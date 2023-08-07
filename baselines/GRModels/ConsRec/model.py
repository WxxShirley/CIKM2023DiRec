import torch.nn as nn
import torch


class ConsRec4RGI(nn.Module):
    def __init__(self, num_users, num_items, num_groups, user_hg_train, user_hg_val, gi_graph, gg_graph, gg_graph_val,
                 emb_dim, layers):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_groups = num_groups

        self.user_hg_train = user_hg_train
        self.user_hg_val = user_hg_val

        self.gi_graph = gi_graph

        self.gg_graph = gg_graph
        self.gg_graph_val = gg_graph_val

        self.emb_dim = emb_dim
        self.layers = layers

        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.item_embedding = nn.Embedding(num_items, self.emb_dim)
        self.group_embedding = nn.Embedding(num_groups, self.emb_dim)

        self.act = nn.Sigmoid()

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.group_embedding.weight)

        self.hyper_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())
        self.lightgcn_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())
        self.overlap_gate = nn.Sequential(nn.Linear(self.emb_dim, 1), nn.Sigmoid())

    def compute(self, mode="val"):
        # user-hg
        hg_embeds = torch.cat([self.user_embedding.weight, self.group_embedding.weight], dim=0)
        all_hg_embeds = [hg_embeds]

        user_hg = self.user_hg_train if mode == "val" else self.user_hg_val

        for _ in range(self.layers):
            hg_embeds = torch.sparse.mm(user_hg, hg_embeds)
            all_hg_embeds.append(hg_embeds)

        hg_emb = torch.mean(torch.stack(all_hg_embeds), dim=0)
        hg_users, hg_groups = torch.split(hg_emb, [self.num_users, self.num_groups], dim=0)

        # gi
        gi_embeds = torch.cat([self.group_embedding.weight, self.item_embedding.weight], dim=0)

        for _ in range(1):
            gi_embeds = torch.sparse.mm(self.gi_graph, gi_embeds)
        gi_groups, _ = torch.split(gi_embeds, [self.num_groups, self.num_items])

        # gg
        gg_embeds = self.group_embedding.weight

        gg_graph = self.gg_graph if mode == "val" else self.gg_graph_val
        gg_embeds = torch.mm(gg_graph, gg_embeds)

        hyper_coef = self.hyper_gate(hg_groups)
        lightgcn_coef = self.lightgcn_gate(gi_groups)
        overlap_coef = self.overlap_gate(gg_embeds)

        group_emb_final = hyper_coef * hg_groups + lightgcn_coef * gi_groups + overlap_coef * gg_embeds
        user_emb_final = hg_users

        return user_emb_final, group_emb_final

    def forward(self, user_inputs, pos_groups, neg_groups):
        all_users, all_groups = self.compute()

        user_embeds = all_users[user_inputs]
        pos_embeds = all_groups[pos_groups]
        neg_embeds = all_groups[neg_groups]

        user_emb_ego = self.user_embedding(user_inputs)
        pos_emb_ego = self.group_embedding(pos_groups)
        neg_emb_ego = self.group_embedding(neg_groups)

        return user_embeds, pos_embeds, neg_embeds, user_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, user_inputs, pos_groups, neg_groups):
        (user_embeds, pos_embeds, neg_embeds, user_emb_0, pos_emb_0, neg_emb_0) = self.forward(user_inputs, pos_groups,
                                                                                               neg_groups)

        reg_loss = (1 / 2) * (user_emb_0.norm(2).pow(2) + pos_emb_0.norm(2).pow(2) + neg_emb_0.norm(2).pow(2)) / float(
            len(user_inputs))

        pos_score = torch.sum(user_embeds * pos_embeds, dim=-1)
        neg_score = torch.sum(user_embeds * neg_embeds, dim=-1)

        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_score - pos_score))

        return bpr_loss, reg_loss

    def get_user_rating(self, mode="val"):
        all_users, all_groups = self.compute(mode=mode)

        rating = self.act(torch.mm(all_users, all_groups.t()))
        # rating = torch.mm(all_users, all_groups.t())
        return rating
