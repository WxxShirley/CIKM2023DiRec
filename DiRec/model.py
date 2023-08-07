import torch
import torch.nn as nn
from torch_scatter import scatter
import torch.nn.functional as F


class CollaborativeGCNConv(nn.Module):
    def __init__(self, n_layer, n_nodes):
        super(CollaborativeGCNConv, self).__init__()
        self.n_layer = n_layer
        self.n_nodes = n_nodes

    def forward(self, embed, edge_index, trend, return_final=False):
        agg_embed = embed
        all_embeds = [embed]

        row, col = edge_index

        for _ in range(self.n_layer):
            out = agg_embed[row] * trend.unsqueeze(-1)
            agg_embed = scatter(out, col, dim=0, dim_size=self.n_nodes, reduce='add')
            all_embeds.append(agg_embed)

        if return_final:
            return agg_embed

        return torch.mean(torch.stack(all_embeds, dim=0), dim=0)


class DualIntentRecModel(nn.Module):
    def __init__(self, num_user, num_item, num_group, emb_dim, layer, hg_dict, user_side_ssl, group_side_ssl):
        super(DualIntentRecModel, self).__init__()

        self.num_user = num_user
        self.num_item = num_item
        self.num_group = num_group

        self.emb_dim = emb_dim
        self.layer = layer

        # Use when model training and testing on validation set
        self.user_hg_train = hg_dict["UserHGTrain"]
        # Use when model testing on test set
        self.user_hg_val = hg_dict["UserHGVal"]
        self.item_hg = hg_dict["ItemHG"]
        self.ui_trend, self.ui_edge = hg_dict["UITrend"], hg_dict["UIEdge"]

        self.user_side_ssl = user_side_ssl
        self.group_side_ssl = group_side_ssl

        self.user_embedding_distinct = nn.Embedding(num_user, emb_dim)
        self.user_embedding_interest = nn.Embedding(num_user, emb_dim)
        self.group_embedding_distinct = nn.Embedding(num_group, emb_dim)
        self.group_embedding_interest = nn.Embedding(num_group, emb_dim)
        self.item_embedding = nn.Embedding(num_item, emb_dim)

        self.act = nn.Sigmoid()
        self.cgcn = CollaborativeGCNConv(1, num_user + num_item)

        nn.init.normal_(self.user_embedding_distinct.weight, std=0.1)
        nn.init.normal_(self.user_embedding_interest.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        nn.init.normal_(self.group_embedding_distinct.weight, std=0.1)
        nn.init.normal_(self.group_embedding_interest.weight, std=0.1)

    def compute(self, test_on_testset=False, return_other=False):
        """Forward Propagation"""
        user_hg = self.user_hg_train
        if test_on_testset:
            user_hg = self.user_hg_val
        ug_emb = torch.cat([self.user_embedding_distinct.weight, self.group_embedding_distinct.weight], dim=0)
        for _ in range(self.layer):
            ug_emb = torch.sparse.mm(user_hg, ug_emb)
        user_emb, group_emb_u_side = torch.split(ug_emb, [self.num_user, self.num_group], dim=0)

        # Ablation - Social Intent
        # return user_emb, group_emb_u_side

        ig_emb = torch.cat([self.item_embedding.weight, self.group_embedding_interest.weight], dim=0)
        for _ in range(self.layer):
            ig_emb = torch.sparse.mm(self.item_hg, ig_emb)
        _, group_emb_i_side = torch.split(ig_emb, [self.num_item, self.num_group], dim=0)

        ui_emb = torch.cat([self.user_embedding_interest.weight, self.item_embedding.weight], dim=0)
        ui_emb_final = self.cgcn(ui_emb, self.ui_edge, self.ui_trend)
        user_emb_i_side, _ = torch.split(ui_emb_final, [self.num_user, self.num_item], dim=0)

        # Ablation - Interest Intent
        # return user_emb_i_side, group_emb_i_side

        final_u_emb = torch.cat([user_emb, user_emb_i_side], dim=1)
        final_g_emb = torch.cat([group_emb_u_side, group_emb_i_side], dim=1)

        if return_other:
            return final_u_emb, final_g_emb, user_emb_i_side, user_emb, group_emb_i_side, group_emb_u_side

        return final_u_emb, final_g_emb

    def forward(self, user_inputs, pos_groups, neg_groups):
        all_users, all_groups, all_user_emb_i_side, all_user_emb_g_side, all_group_emb_i_side, all_group_emb_u_side = self.compute(
            return_other=True)
        # all_users, all_groups = self.compute(return_other=True)

        user_embeds = all_users[user_inputs]
        pos_embed = all_groups[pos_groups]
        neg_embed = all_groups[neg_groups]

        user_embeds_ego1, user_embeds_ego2 = self.user_embedding_interest(user_inputs), self.user_embedding_distinct(
            user_inputs)
        pos_embed_ego1, pos_emb_ego2 = self.group_embedding_interest(pos_groups), self.group_embedding_distinct(
            pos_groups)
        neg_embed_ego1, neg_emb_ego2 = self.group_embedding_interest(neg_groups), self.group_embedding_distinct(
            neg_groups)

        reg_loss = (1 / 2) * (
                user_embeds_ego1.norm(2).pow(2) + user_embeds_ego2.norm(2).pow(2) + pos_embed_ego1.norm(2).pow(
            2) + pos_emb_ego2.norm(2).pow(2) + neg_embed_ego1.norm(2).pow(2) + neg_emb_ego2.norm(2).pow(2)) \
                   / float(len(user_inputs))

        ssl_loss = None
        if self.user_side_ssl:
            ssl_loss = self.ssl_loss(all_user_emb_i_side[user_inputs], all_user_emb_g_side[user_inputs],
                                     all_user_emb_g_side) + self.ssl_loss(all_user_emb_g_side[user_inputs],
                                                                          all_user_emb_i_side[user_inputs],
                                                                          all_user_emb_i_side)
        if self.group_side_ssl:
            group_loss = self.ssl_loss(all_group_emb_i_side[pos_groups], all_group_emb_u_side[pos_groups],
                                       all_group_emb_u_side) + self.ssl_loss(all_group_emb_u_side[pos_groups],
                                                                             all_group_emb_i_side[pos_groups],
                                                                             all_group_emb_i_side)
            if ssl_loss is not None:
                ssl_loss += group_loss
            else:
                ssl_loss = group_loss

        return user_embeds, pos_embed, neg_embed, reg_loss, ssl_loss

    def bpr_loss(self, user_input, pos_group_input, neg_group_input):
        """Loss computation, including both BPR loss and SSL loss"""
        (user_emb, pos_emb, neg_emb, reg_loss, twin_loss) = self.forward(user_input, pos_group_input, neg_group_input)

        pos_score = torch.sum(user_emb * pos_emb, dim=-1)
        neg_score = torch.sum(user_emb * neg_emb, dim=-1)
        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_score - pos_score))

        return bpr_loss, reg_loss, twin_loss

    def get_user_rating(self, mode="val"):
        if mode == "val":
            all_users, all_groups = self.compute()
        elif mode == "test":
            all_users, all_groups = self.compute(test_on_testset=True)

        rating = self.act(torch.mm(all_users, all_groups.t()))
        # rating = torch.mm(all_users, all_groups.t())
        return rating

    def ssl_loss(self, user_emb_side1, user_emb_side2, user_emb_side2_all):
        norm_user_side1 = F.normalize(user_emb_side1)
        norm_user_side2 = F.normalize(user_emb_side2)
        norm_all_user_side2 = F.normalize(user_emb_side2_all)

        pos_score = torch.mul(norm_user_side1, norm_user_side2).sum(dim=1)
        total_score = torch.matmul(norm_user_side1, norm_all_user_side2.transpose(0, 1))

        pos_score = torch.exp(pos_score / 0.1)
        total_score = torch.exp(total_score / 0.1).sum(dim=1)
        ssl_loss = -torch.log(pos_score / total_score).sum()

        return ssl_loss / user_emb_side1.shape[0]
