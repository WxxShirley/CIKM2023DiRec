import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse


class RecModel4RGI(nn.Module):
    def __init__(self, num_user, num_group, num_layer, emb_dim, adj, adj_val, model_type="MF", n_intents=16):
        super().__init__()

        self.num_user = num_user
        self.num_group = num_group

        self.num_layer = num_layer
        self.emb_dim = emb_dim

        # "MF" / "NGCF" / "LightGCN" / "SGL" / "DCCF"
        self.model_type = model_type
        self.adj = adj
        self.adj_val = adj_val
        self.device = adj.device

        self.user_embedding = nn.Embedding(num_user, emb_dim)
        self.group_embedding = nn.Embedding(num_group, emb_dim)

        if self.model_type == "NGCF":
            self.gc = nn.Linear(self.emb_dim, self.emb_dim, bias=True)
            self.bi = nn.Linear(self.emb_dim, self.emb_dim, bias=True)

        if self.model_type == "DCCF":
            self.all_h_list = adj.coalesce().indices()[0]
            self.all_t_list = adj.coalesce().indices()[1]
            self.A_shape = adj.shape

            # prepare intents
            _user_intent = torch.empty(self.emb_dim, n_intents)
            nn.init.xavier_normal_(_user_intent)
            self.user_intent = torch.nn.Parameter(_user_intent, requires_grad=True)

            _item_intent = torch.empty(self.emb_dim, n_intents)
            nn.init.xavier_normal_(_item_intent)
            self.item_intent = torch.nn.Parameter(_item_intent, requires_grad=True)

        self.act = nn.Sigmoid()

        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.group_embedding.weight)
        # nn.init.normal_(self.user_embedding.weight, std=0.1)
        # nn.init.normal_(self.group_embedding.weight, std=0.1)

    def simgcl_compute(self):
        """Computation used in **SimGCL(SIGIR'22)** - Randomly adding noises to embeddings"""
        all_emb = torch.cat([self.user_embedding.weight, self.group_embedding.weight], dim=0)
        embeddings = [all_emb]

        device = all_emb.device

        for _ in range(self.num_layer):
            all_emb = torch.sparse.mm(self.adj, all_emb)
            # add random noise
            random_noise = torch.empty(all_emb.shape).uniform_().to(device)
            all_emb += torch.mul(torch.sign(all_emb), torch.nn.functional.normalize(random_noise, p=2, dim=1)) * 0.2
            embeddings.append(all_emb)

        output = torch.mean(torch.stack(embeddings), dim=0)
        users, groups = torch.split(output, [self.num_user, self.num_group], dim=0)
        return users, groups

    def compute(self, test_on_testset=False):

        # For MF-BPR, just return the user/group embedding tables
        if self.model_type == "MF":
            return self.user_embedding.weight, self.group_embedding.weight

        # For SGL, LightGCN, and SimGCL, forward propagation
        if self.model_type in ["SGL", "LightGCN", "SimGCL"]:
            adj = self.adj
            if test_on_testset:
                adj = self.adj_val

            user_emb = self.user_embedding.weight
            group_emb = self.group_embedding.weight

            all_emb = torch.cat([user_emb, group_emb])
            embeddings = [all_emb]

            for _ in range(self.num_layer):
                all_emb = torch.sparse.mm(adj, all_emb)
                embeddings.append(all_emb)

            output = torch.mean(torch.stack(embeddings, dim=0), dim=0)
            users, groups = torch.split(output, [self.num_user, self.num_group], dim=0)
            return users, groups

        # NGCF has a specfic convlutional mechanism
        if self.model_type == "NGCF":
            adj = self.adj
            if test_on_testset:
                adj = self.adj_val

            user_emb = self.user_embedding.weight
            group_emb = self.group_embedding.weight

            ego_emb = torch.cat([user_emb, group_emb])
            all_embs = [ego_emb]

            for _ in range(self.num_layer):
                side_embed = torch.sparse.mm(adj, ego_emb)
                sum_embed = self.gc(side_embed)

                bi_emb = torch.mul(ego_emb, side_embed)
                bi_emb = self.bi(bi_emb)

                ego_emb = F.leaky_relu(sum_embed + bi_emb)
                norm_embed = F.normalize(ego_emb, p=2, dim=1)
                all_embs.append(norm_embed)

            output = torch.mean(torch.stack(all_embs, dim=0), dim=0)
            users, groups = torch.split(output, [self.num_user, self.num_group], dim=0)
            return users, groups

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
        if self.model_type == "DCCF":
            all_users, all_groups = self.dccf_compute(test_on_testset=mode == "test")
        else:
            if mode == "val":
                all_users, all_groups = self.compute()
            elif mode == "test":
                all_users, all_groups = self.compute(test_on_testset=True)

        rating = self.act(torch.mm(all_users, all_groups.t()))
        # rating = torch.mm(all_users, all_groups.t())
        return rating

    def sgl_compute(self, graph):
        """Computation used in **SGL(SIGIR21)** - Forward propagation via a corrupted graph"""
        user_emb, group_emb = self.user_embedding.weight, self.group_embedding.weight
        all_emb = torch.cat([user_emb, group_emb])

        embeddings = [all_emb]

        for _ in range(self.num_layer):
            all_emb = torch.sparse.mm(graph, all_emb)
            embeddings.append(all_emb)

        output = torch.mean(torch.stack(embeddings, dim=0), dim=0)
        users, groups = torch.split(output, [self.num_user, self.num_group])
        return users, groups

    def simgcl_loss(self, user_input, pos_input):
        """Contrastive loss used for SimGCL model"""
        all_user_emb1, all_group_emb1 = self.simgcl_compute()
        all_user_emb2, all_group_emb2 = self.simgcl_compute()

        all_user_emb1 = F.normalize(all_user_emb1, dim=1)
        all_user_emb2 = F.normalize(all_user_emb2, dim=1)
        all_group_emb1 = F.normalize(all_group_emb1, dim=1)
        all_group_emb2 = F.normalize(all_group_emb2, dim=1)

        user_emb1, user_emb2 = all_user_emb1[user_input], all_user_emb2[user_input]
        group_emb1, group_emb2 = all_group_emb1[pos_input], all_group_emb2[pos_input]

        pos_rating_user = torch.sum(user_emb1 * user_emb2, dim=-1)
        tot_rating_user = torch.matmul(user_emb1, torch.transpose(all_user_emb2, 0, 1))
        ssl_logits_user = tot_rating_user - pos_rating_user[:, None]
        logit_user = torch.logsumexp(ssl_logits_user / 0.1, dim=1)

        pos_rating_group = torch.sum(group_emb1 * group_emb2, dim=-1)
        tot_rating_group = torch.matmul(group_emb1, torch.transpose(all_group_emb2, 0, 1))
        ssl_logits_group = tot_rating_group - pos_rating_group[:, None]
        logit_group = torch.logsumexp(ssl_logits_group / 0.1, dim=1)

        return torch.sum(logit_user + logit_group) / ssl_logits_user.shape[0] / 2

    def ssl_loss(self, subgraph1, subgraph2, user_input, pos_input):
        all_user_emb1, all_group_emb1 = self.sgl_compute(subgraph1)
        all_user_emb2, all_group_emb2 = self.sgl_compute(subgraph2)

        all_user_emb1 = F.normalize(all_user_emb1, dim=1)
        all_user_emb2 = F.normalize(all_user_emb2, dim=1)
        all_group_emb1 = F.normalize(all_group_emb1, dim=1)
        all_group_emb2 = F.normalize(all_group_emb2, dim=1)

        user_emb1, user_emb2 = all_user_emb1[user_input], all_user_emb2[user_input]
        group_emb1, group_emb2 = all_group_emb1[pos_input], all_group_emb2[pos_input]

        pos_rating_user = torch.sum(user_emb1 * user_emb2, dim=-1)
        tot_rating_user = torch.matmul(user_emb1, torch.transpose(all_user_emb2, 0, 1))
        ssl_logits_user = tot_rating_user - pos_rating_user[:, None]
        logit_user = torch.logsumexp(ssl_logits_user / 0.1, dim=1)

        pos_rating_group = torch.sum(group_emb1 * group_emb2, dim=-1)
        tot_rating_group = torch.matmul(group_emb1, torch.transpose(all_group_emb2, 0, 1))
        ssl_logits_group = tot_rating_group - pos_rating_group[:, None]
        logit_group = torch.logsumexp(ssl_logits_group / 0.1, dim=1)

        return torch.sum(logit_user + logit_group) / ssl_logits_user.shape[0] / 2

    def adaptive_mask_(self, head_embeddings, tail_embeddings):
        # Adaptive mask for DCCF model

        # step1: normalization
        head_embeddings = torch.nn.functional.normalize(head_embeddings)
        tail_embeddings = torch.nn.functional.normalize(tail_embeddings)

        # step2: compute adaptive augmented edge weight
        edge_alpha = (torch.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2

        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=edge_alpha,
                                             sparse_sizes=self.A_shape).to(self.device)
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)

        G_indices = torch.stack([self.all_h_list, self.all_t_list], dim=0)
        # step3: degree normalization
        G_values = D_scores_inv[self.all_h_list] * edge_alpha

        # return augmented adj
        return G_indices, G_values

    def dccf_compute(self, test_on_testset=False, compute_ssl_loss=False):
        all_embeddings = [torch.cat([self.user_embedding.weight, self.group_embedding.weight], dim=0)]

        adj = self.adj

        if test_on_testset:
            adj = self.adj_val

        # for ssl loss
        gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings = [], [], [], []

        for i in range(self.num_layer):
            # GNN propagate
            gnn_layer_embeddings = torch.sparse.mm(adj, all_embeddings[i])

            # Intent propagate
            u_emb, i_emb = torch.split(all_embeddings[i], [self.num_user, self.num_group])
            u_intent_embeddings = torch.softmax(u_emb @ self.user_intent, dim=1) @ self.user_intent.T
            i_intent_embeddings = torch.softmax(i_emb @ self.item_intent, dim=1) @ self.item_intent.T
            int_layer_embeddings = torch.cat([u_intent_embeddings, i_intent_embeddings], dim=0)

            # Intent augment propagate
            int_head_embeddings = torch.index_select(int_layer_embeddings, 0, self.all_h_list)
            int_tail_embeddings = torch.index_select(int_layer_embeddings, 0, self.all_t_list)
            int_g_indvices, int_g_values = self.adaptive_mask_(int_head_embeddings, int_tail_embeddings)
            intent_augment_layer_embeddings = torch_sparse.spmm(int_g_indvices, int_g_values, self.A_shape[0],
                                                                self.A_shape[1], all_embeddings[i])

            # GNN augment propagate
            gnn_head_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.all_h_list)
            gnn_tail_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.all_t_list)
            gnn_g_indices, gnn_g_values = self.adaptive_mask_(gnn_head_embeddings, gnn_tail_embeddings)
            gnn_augment_layer_embeddings = torch_sparse.spmm(gnn_g_indices, gnn_g_values, self.A_shape[0],
                                                             self.A_shape[1], all_embeddings[i])

            all_embeddings.append(
                gnn_layer_embeddings + int_layer_embeddings + intent_augment_layer_embeddings + gnn_augment_layer_embeddings)
            gnn_embeddings.append(gnn_layer_embeddings)
            int_embeddings.append(int_layer_embeddings)
            gaa_embeddings.append(gnn_augment_layer_embeddings)
            iaa_embeddings.append(intent_augment_layer_embeddings)

        all_embeddings = torch.mean(torch.stack(all_embeddings, dim=1), dim=1, keepdim=False)
        user_embeds, group_embeds = torch.split(all_embeddings, [self.num_user, self.num_group], dim=0)

        if compute_ssl_loss:
            return user_embeds, group_embeds, gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings

        return user_embeds, group_embeds

    def dccf_forward(self, users, pos_items, neg_items):
        ua_embedding, ga_embedding, gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings = self.dccf_compute(
            compute_ssl_loss=True)

        u_emb = ua_embedding[users]
        pos_emb = ga_embedding[pos_items]
        neg_emb = ga_embedding[neg_items]

        pos_scores = torch.sum(u_emb * pos_emb, dim=-1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=-1)

        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        u_emb_ego = self.user_embedding(users)
        pos_emb_ego = self.group_embedding(pos_items)
        neg_emb_ego = self.group_embedding(neg_items)

        reg_loss = (1 / 2) * (
                    u_emb_ego.norm(2).pow(2) + pos_emb_ego.norm(2).pow(2) + neg_emb_ego.norm(2).pow(2)) / float(
            len(users))

        int_loss = (self.user_intent.norm(2).pow(2) + self.item_intent.norm(2).pow(2)) / self.user_intent.shape[0]

        ssl_loss = self.cal_dccf_ssl_loss(users, pos_items, gnn_embeddings, int_embeddings, gaa_embeddings,
                                          iaa_embeddings)

        return bpr_loss, reg_loss, int_loss, ssl_loss

    def cal_dccf_ssl_loss(self, users, items, gnn_emb, int_emb, gaa_emb, iaa_emb):
        users = torch.unique(users)
        items = torch.unique(items)

        cl_loss = 0.0

        def cal_loss(emb1, emb2):
            pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / 0.1)
            neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / 0.1), axis=1)
            loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
            loss /= pos_score.shape[0]
            return loss

        for i in range(len(gnn_emb)):
            u_gnn_embs, i_gnn_embs = torch.split(gnn_emb[i], [self.num_user, self.num_group], 0)
            u_int_embs, i_int_embs = torch.split(int_emb[i], [self.num_user, self.num_group], 0)
            u_gaa_embs, i_gaa_embs = torch.split(gaa_emb[i], [self.num_user, self.num_group], 0)
            u_iaa_embs, i_iaa_embs = torch.split(iaa_emb[i], [self.num_user, self.num_group], 0)

            u_gnn_embs = F.normalize(u_gnn_embs[users], dim=1)
            u_int_embs = F.normalize(u_int_embs[users], dim=1)
            u_gaa_embs = F.normalize(u_gaa_embs[users], dim=1)
            u_iaa_embs = F.normalize(u_iaa_embs[users], dim=1)

            i_gnn_embs = F.normalize(i_gnn_embs[items], dim=1)
            i_int_embs = F.normalize(i_int_embs[items], dim=1)
            i_gaa_embs = F.normalize(i_gaa_embs[items], dim=1)
            i_iaa_embs = F.normalize(i_iaa_embs[items], dim=1)

            cl_loss += cal_loss(u_gnn_embs, u_int_embs)
            cl_loss += cal_loss(u_gnn_embs, u_gaa_embs)
            cl_loss += cal_loss(u_gnn_embs, u_iaa_embs)

            cl_loss += cal_loss(i_gnn_embs, i_int_embs)
            cl_loss += cal_loss(i_gnn_embs, i_gaa_embs)
            cl_loss += cal_loss(i_gnn_embs, i_iaa_embs)

        return cl_loss
