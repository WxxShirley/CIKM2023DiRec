import torch
import torch.nn as nn


class CFGA(nn.Module):
    def __init__(self, num_user, num_item, num_group, num_layer, emb_dim, norm_adj, norm_adj_val, norm_adj_ui,
                 norm_adj_gi, R, R_val, R_ui, att_coef, contexual_emb_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_user, emb_dim)
        self.item_embedding = nn.Embedding(num_item, emb_dim)
        self.group_embedding = nn.Embedding(num_group, emb_dim)
        self.num_user, self.num_item, self.num_group = num_user, num_item, num_group

        self.contextual_item_embedding = nn.Embedding(num_item, contexual_emb_dim)
        self.contextual_group_embedding = nn.Embedding(num_group, contexual_emb_dim)

        self.num_layer = num_layer
        self.attn_coef = att_coef

        # Three adj matrices for forward propagation
        self.norm_adj = norm_adj
        self.norm_adj_val = norm_adj_val
        self.norm_adj_ui = norm_adj_ui
        self.norm_adj_gi = norm_adj_gi
        self.R = R
        self.R_val = R_val
        self.R_ui = R_ui

        self.act = nn.Sigmoid()

        self.emb_dim = emb_dim

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.group_embedding.weight, std=0.01)
        nn.init.normal_(self.contextual_group_embedding.weight, std=0.01)
        nn.init.normal_(self.contextual_item_embedding.weight, std=0.01)

    def contextual_graph_attention(self, target_embedding, contextual_embedding, compute_type="gu",
                                   test_on_testset=False):
        sim_score = torch.mm(contextual_embedding, contextual_embedding.T)
        sim_score = torch.softmax(sim_score, dim=1)

        # TODO: check dim
        compute_adj = self.R
        if compute_type == "gu" and test_on_testset:
            compute_adj = self.R_val
        if compute_type == "iu":
            compute_adj = self.R_ui

        # attention_matrix = torch.softmax(torch.sparse.mm(compute_adj, sim_score), dim=0)
        attention_matrix = torch.softmax(nn.LeakyReLU()(torch.sparse.mm(compute_adj, sim_score)), dim=1)

        attention_emb = torch.mm(attention_matrix, target_embedding)
        # print(attention_emb[:5, :8])
        return attention_emb

    def compute(self, test_on_testset=False):
        # Partition Layer
        group_emb_user_part, group_emb_item_part = self.group_embedding.weight, self.group_embedding.weight
        user_emb_group_part, user_emb_item_part = self.user_embedding.weight, self.user_embedding.weight
        item_emb_user_part, item_emb_group_part = self.item_embedding.weight, self.item_embedding.weight

        group_emb_user_parts, group_emb_item_parts = [group_emb_user_part], [group_emb_item_part]
        user_emb_group_parts, user_emb_item_parts = [user_emb_group_part], [user_emb_item_part]

        # print(user_emb_group_part[:5, :5])

        norm_adj = self.norm_adj
        if test_on_testset:
            norm_adj = self.norm_adj_val

        # Aggregation Layer
        for _ in range(self.num_layer):
            # Augmentation Layer
            group2user_attention_emb = self.contextual_graph_attention(group_emb_user_part,
                                                                       self.contextual_group_embedding.weight,
                                                                       test_on_testset=test_on_testset)
            item2user_attention_emb = self.contextual_graph_attention(item_emb_user_part,
                                                                      self.contextual_item_embedding.weight,
                                                                      compute_type="iu",
                                                                      test_on_testset=test_on_testset)

            user_emb_group_part = user_emb_group_part + self.attn_coef * group2user_attention_emb
            user_emb_group_part = nn.functional.normalize(user_emb_group_part, dim=1)
            user_emb_item_part = user_emb_item_part + self.attn_coef * item2user_attention_emb
            user_emb_item_part = nn.functional.normalize(user_emb_item_part, dim=1)

            # # U-G forward propagation
            ug_graph_embeds = torch.sparse.mm(norm_adj, torch.cat([user_emb_group_part, group_emb_user_part], dim=0))
            user_emb_group_part, group_emb_user_part = torch.split(ug_graph_embeds, [self.num_user, self.num_group])
            user_emb_group_parts.append(user_emb_group_part)
            group_emb_user_parts.append(group_emb_user_part)
            # print(user_emb_group_part[:5, :5])

            # U-I forward propagation
            ui_graph_embeds = torch.sparse.mm(self.norm_adj_ui,
                                              torch.cat([user_emb_item_part, item_emb_user_part], dim=0))
            user_emb_item_part, item_emb_user_part = torch.split(ui_graph_embeds, [self.num_user, self.num_item])
            user_emb_item_parts.append(user_emb_item_part)

            # G-I forward propagation
            gi_graph_embeds = torch.sparse.mm(self.norm_adj_gi,
                                              torch.cat([group_emb_item_part, item_emb_group_part], dim=0))
            group_emb_item_part, item_emb_group_part = torch.split(gi_graph_embeds, [self.num_group, self.num_item])
            group_emb_item_parts.append(group_emb_item_part)

        # Merging Layer
        user_emb_group_part_final, user_emb_item_part_final = torch.mean(torch.stack(user_emb_group_parts),
                                                                         dim=0), torch.mean(
            torch.stack(user_emb_item_parts), dim=0)
        user_embedding = torch.cat([user_emb_group_part_final, user_emb_item_part_final], dim=1)

        group_emb_user_part_final, group_emb_item_part_final = torch.mean(torch.stack(group_emb_user_parts),
                                                                          dim=0), torch.mean(
            torch.stack(group_emb_item_parts), dim=0)
        group_embedding = torch.cat([group_emb_user_part_final, group_emb_item_part_final], dim=1)

        return user_embedding, group_embedding

    def forward(self, user_inputs, pos_groups, neg_groups):
        all_users, all_groups = self.compute()

        user_embeds = all_users[user_inputs]
        pos_embed = all_groups[pos_groups]
        neg_embed = all_groups[neg_groups]

        user_embeds_ego = self.user_embedding(user_inputs)
        pos_embed_ego = self.group_embedding(pos_groups)
        neg_embed_ego = self.group_embedding(neg_groups)

        return user_embeds, pos_embed, neg_embed, user_embeds_ego, pos_embed_ego, neg_embed_ego

    def bpr_loss(self, user_input, pos_group_input, neg_group_input):
        """Loss computation"""
        (user_emb, pos_emb, neg_emb, user_emb_0, pos_emb_0, neg_emb_0) = self.forward(user_input, pos_group_input,
                                                                                      neg_group_input)

        # regularization loss
        reg_loss = (user_emb_0.norm(2).pow(2) + pos_emb_0.norm(2).pow(2) + neg_emb_0.norm(2).pow(2)) \
                   / float(len(user_input))

        # bpr loss
        pos_score = torch.sum(user_emb * pos_emb, dim=-1)
        neg_score = torch.sum(user_emb * neg_emb, dim=-1)
        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_score - pos_score))

        return bpr_loss, reg_loss

    def get_user_rating(self, mode="val"):
        test_mode = mode == "test"
        all_users, all_groups = self.compute(test_on_testset=test_mode)
        rating = self.act(torch.mm(all_users, all_groups.t()))
        # rating = torch.mm(all_users, all_groups.t())
        return rating
