from collections import defaultdict
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import datautil
import torch


class Data(object):
    def __init__(self, dataset_name="Mafengwo", batch_size=1024, num_negatives=6):
        print(f"Loading [{dataset_name.upper()}] dataset ... ")

        self.dataset = dataset_name
        self.num_negatives = num_negatives
        self.batch_size = batch_size

        self.ug_train_dict = datautil.load_file_to_dict_format(f"../../../data/{dataset_name}/train.txt")
        self.ug_test_dict = datautil.load_file_to_dict_format(f"../../../data/{dataset_name}/test.txt")
        self.ug_val_dict = datautil.load_file_to_dict_format(f"../../../data/{dataset_name}/val.txt")

        self.n_users, self.n_groups, self.max_member_len = 0, 0, 0
        self.train_gu_dict, self.val_gu_dict = defaultdict(list), defaultdict(list)

        self.train_group_users, self.train_group_masks = None, None
        self.val_group_users, self.val_group_masks = None, None
        self.prepare_data()

    def prepare_data(self):
        train_ug_u, train_ug_g = datautil.load_file_to_list_format(f"../../../data/{self.dataset}/train.txt")
        val_ug_u, val_ug_g = datautil.load_file_to_list_format(f"../../../data/{self.dataset}/val.txt")

        self.n_users = max(max(train_ug_u), max(val_ug_u)) + 1
        self.n_groups = max(max(train_ug_g), max(val_ug_g)) + 1

        for u, g in zip(train_ug_u, train_ug_g):
            self.train_gu_dict[g].append(u)

        for u, g in zip(train_ug_u + val_ug_u, train_ug_g + val_ug_g):
            self.val_gu_dict[g].append(u)

        self.max_member_len = max([len(members) for members in self.val_gu_dict.values()]) + 1
        # print(self.max_member_len)

        # train
        all_group_masks, all_group_users = [], []
        for g in range(self.n_groups):
            # 把group也作为一个成员
            members = self.train_gu_dict[g] + [g + self.n_users]
            group_mask = torch.zeros((self.max_member_len,))
            group_mask[len(members):] = -np.inf
            if len(members) < self.max_member_len:
                group_user = torch.hstack(
                    (torch.LongTensor(members), torch.zeros((self.max_member_len - len(members),)))).long()
            else:
                group_user = torch.LongTensor(members)

            all_group_masks.append(group_mask)
            all_group_users.append(group_user)

        self.train_group_masks = torch.stack(all_group_masks)
        self.train_group_users = torch.stack(all_group_users)
        # print(self.train_group_masks[80: 100,:], self.train_group_users[80: 100,:])
        # print(train_group_masks.shape, train_group_users.shape)

        # test
        all_group_masks, all_group_users = [], []
        for g in range(self.n_groups):
            members = self.val_gu_dict[g] + [g + self.n_users]
            group_mask = torch.zeros((self.max_member_len,))
            group_mask[len(members):] = -np.inf
            if len(members) < self.max_member_len:
                group_user = torch.hstack(
                    (torch.LongTensor(members), torch.zeros((self.max_member_len - len(members),)))).long()
            else:
                group_user = torch.LongTensor(members)
            all_group_users.append(group_user)
            all_group_masks.append(group_mask)

        self.val_group_masks = torch.stack(all_group_masks)
        self.val_group_users = torch.stack(all_group_users)

    def get_train_instances(self):
        users, pos_groups, neg_groups = [], [], []

        for u, groups in self.ug_train_dict.items():
            for g in groups:
                users.extend([u] * self.num_negatives)
                pos_groups.extend([g] * self.num_negatives)

                for _ in range(self.num_negatives):
                    neg_group = np.random.randint(self.n_groups)
                    while neg_group in groups:
                        neg_group = np.random.randint(self.n_groups)
                    neg_groups.append(neg_group)

        return users, pos_groups, neg_groups

    def get_user_dataloader(self, batch_size=512):
        users, pos_groups, neg_groups = self.get_train_instances()
        train_data = TensorDataset(torch.LongTensor(users), torch.LongTensor(pos_groups), torch.LongTensor(neg_groups))
        return DataLoader(train_data, batch_size=batch_size, shuffle=True)


# Test Code
# data = Data(dataset_name="Mafengwo")
# data = Data(dataset_name="Weeplaces")
# data = Data(dataset_name="Steam")
