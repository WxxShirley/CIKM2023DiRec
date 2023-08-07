import torch
import torch.nn as nn


class SelfAttnPooling(nn.Module):
    def __init__(self, emb_dim, drop_ratio):
        super().__init__()
        self.score_layer = nn.Sequential(
            nn.Linear(emb_dim, 32),
            nn.ReLU(),
            # nn.Dropout(drop_ratio),
            nn.Linear(32, 1)
        )

    def forward(self, x, mask):
        weight = self.score_layer(x)
        weight = torch.softmax(weight + mask.unsqueeze(2), dim=1)
        # print(weight)
        return torch.matmul(x.transpose(2, 1), weight).squeeze(2)


class AGREE4Rec(nn.Module):
    def __init__(self, num_users, num_groups, emb_dim, drop_ratio, train_group_users, train_group_masks,
                 val_group_users, val_group_masks):
        super(AGREE4Rec, self).__init__()

        self.num_users = num_users
        self.num_groups = num_groups

        self.all_embedding = nn.Embedding(num_users + num_groups, emb_dim)
        # self.group_embeddinng = nn.Embedding(num_groups, emb_dim)

        self.aggregator = SelfAttnPooling(emb_dim, drop_ratio)

        self.train_group_users = train_group_users
        self.train_group_masks = train_group_masks
        self.val_group_users = val_group_users
        self.val_group_masks = val_group_masks

        self.act = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def get_user_rating(self, mode="val"):
        all_group_users = self.train_group_users
        all_group_masks = self.train_group_masks

        if mode == "test":
            all_group_users = self.val_group_users
            all_group_masks = self.val_group_masks

        group_embeds = self.aggregator(self.all_embedding(all_group_users), all_group_masks)
        user_embeds, _ = torch.split(self.all_embedding.weight, [self.num_users, self.num_groups])

        rating = self.act(torch.mm(user_embeds, group_embeds.t()))
        return rating

    def forward(self, users, pos_group_users, pos_group_masks, neg_group_users, neg_group_masks):
        user_embed = self.all_embedding(users)

        pos_members_emb = self.all_embedding(pos_group_users)
        pos_group_emb = self.aggregator(pos_members_emb, pos_group_masks)

        # print(pos_group_emb)

        neg_members_emb = self.all_embedding(neg_group_users)
        neg_group_emb = self.aggregator(neg_members_emb, neg_group_masks)

        pos_score = torch.sum(user_embed * pos_group_emb, dim=-1)
        neg_score = torch.sum(user_embed * neg_group_emb, dim=-1)

        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_score - pos_score))

        return bpr_loss
