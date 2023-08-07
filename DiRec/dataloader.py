import numpy as np
from scipy.sparse import csr_matrix
from torch.utils.data import TensorDataset, DataLoader
import datautil
import torch
from collections import defaultdict


class Data(object):
    def __init__(self, dataset_name="Mafengwo", batch_size=1024, num_negatives=4, aug_dict=None):
        print(f"Loading [{dataset_name.upper()}] dataset ... ")
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_negative = num_negatives
        self.aug_dict = aug_dict

        self.ug_train_dict = datautil.load_file_to_dict_format(f"../data/{dataset_name}/train.txt")
        self.ug_test_dict = datautil.load_file_to_dict_format(f"../data/{dataset_name}/test.txt")
        self.ug_val_dict = datautil.load_file_to_dict_format(f"../data/{dataset_name}/val.txt")

        self.n_users, self.n_items, self.n_groups = 0, 0, 0
        self.user_hg_train, self.user_hg_val, self.item_hg = None, None, None
        self.ui_trend, self.ui_edge = None, None
        self.prepare_data()

    def prepare_data(self):
        train_ui_u, train_ui_i = datautil.load_file_to_list_format(f"../data/{self.dataset_name}/groupItemTrain.txt")
        train_gi_g, train_gi_i = datautil.load_file_to_list_format(f"../data/{self.dataset_name}/userItemTrain.txt")
        train_ug_u, train_ug_g = datautil.load_file_to_list_format(f"../data/{self.dataset_name}/train.txt")
        val_ug_u, val_ug_g = datautil.load_file_to_list_format(f"../data/{self.dataset_name}/val.txt")

        self.n_users = max(max(train_ui_u), max(train_ug_u)) + 1
        self.n_items = max(max(train_ui_i), max(train_gi_i)) + 1
        self.n_groups = max(max(train_ug_g), max(train_gi_g)) + 1

        # User-Group Hyper-graph (Train Model)
        #  Shape (num_user+num_group, num_group)
        user_hg_row, user_hg_col = [], []
        group2user_dict = defaultdict(list)
        for u, g in zip(train_ug_u, train_ug_g):
            group2user_dict[g].append(u)

        for g, members in group2user_dict.items():
            user_hg_row.extend(members + [g + self.n_users])
            user_hg_col.extend([g] * (len(members) + 1))

        user_hg = csr_matrix((np.ones(len(user_hg_row)), (np.array(user_hg_row), np.array(user_hg_col))),
                             shape=(self.n_users + self.n_groups, self.n_groups))
        self.user_hg_train = datautil.convert_sp_mat_to_sp_tensor(datautil.compute_hyper_graph_adj(user_hg))
        # print("Train", self.user_hg_train)

        user_hg_row, user_hg_col = [], []
        group2user_dict = defaultdict(list)
        for u, g in zip(train_ug_u + val_ug_u, train_ug_g + val_ug_g):
            group2user_dict[g].append(u)

        for g, members in group2user_dict.items():
            user_hg_row.extend(members + [g + self.n_users])
            user_hg_col.extend([g] * (len(members) + 1))
        user_hg = csr_matrix((np.ones(len(user_hg_row)), (np.array(user_hg_row), np.array(user_hg_col))),
                             shape=(self.n_users + self.n_groups, self.n_groups))
        self.user_hg_val = datautil.convert_sp_mat_to_sp_tensor(datautil.compute_hyper_graph_adj(user_hg))
        # print("Val", self.user_hg_val)

        # Item-Group Hyper-graph (as an augmentation for IG bipartite graph)
        #  Shape (num_item+num_group, num_group)
        item_hg_row, item_hg_col = [], []
        group2item_dict = defaultdict(list)
        for g, i in zip(train_gi_g, train_gi_i):
            group2item_dict[g].append(i)

        for g, items in group2item_dict.items():
            item_hg_row.extend(items + [g + self.n_items])
            item_hg_col.extend([g] * (len(items) + 1))
        item_hg = csr_matrix((np.ones(len(item_hg_row)), (np.array(item_hg_row), np.array(item_hg_col))),
                             shape=(self.n_items + self.n_groups, self.n_groups))
        self.item_hg = datautil.convert_sp_mat_to_sp_tensor(datautil.compute_hyper_graph_adj(item_hg))

        self.ui_trend, self.ui_edge = datautil.compute_customized_interaction_graph(train_ui_u, train_ui_i,
                                                                                    self.n_users, self.n_items,
                                                                                    self.aug_dict)

    def get_train_instances(self):
        users, pos_groups, neg_groups = [], [], []

        for u, groups in self.ug_train_dict.items():
            for g in groups:
                users.extend([u] * self.num_negative)
                pos_groups.extend([g] * self.num_negative)

                for _ in range(self.num_negative):
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
