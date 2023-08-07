import numpy as np
from scipy.sparse import csr_matrix
from torch.utils.data import TensorDataset, DataLoader
import datautil
import torch
from collections import defaultdict


class Data(object):
    def __init__(self, dataset_name="Mafengwo", batch_size=1024, num_negatives=6):
        print(f"Loading [{dataset_name.upper()}] dataset ... ")

        self.dataset = dataset_name
        self.num_negatives = num_negatives
        self.batch_size = batch_size

        self.ug_train_dict = datautil.load_file_to_dict_format(f"../../../data/{dataset_name}/train.txt")
        self.ug_test_dict = datautil.load_file_to_dict_format(f"../../../data/{dataset_name}/test.txt")
        self.ug_val_dict = datautil.load_file_to_dict_format(f"../../../data/{dataset_name}/val.txt")

        self.n_users, self.n_groups, self.n_items = 0, 0, 0
        self.user_hg, self.user_hg_val = None, None
        self.gi_graph = None
        self.gg_graph_train, self.gg_graph_val = None, None
        self.prepare_data()

    def prepare_data(self):
        # load interaction data
        train_ug_u, train_ug_g = datautil.load_file_to_list_format(f"../../../data/{self.dataset}/train.txt")
        val_ug_u, val_ug_g = datautil.load_file_to_list_format(f"../../../data/{self.dataset}/val.txt")
        train_gi_g, train_gi_i = datautil.load_file_to_list_format(f"../../../data/{self.dataset}/userItemTrain.txt")
        train_ui_u, train_ui_i = datautil.load_file_to_list_format(f"../../../data/{self.dataset}/groupItemTrain.txt")

        self.n_users = max(max(train_ug_u), max(val_ug_u)) + 1
        self.n_groups = max(max(train_ug_g), max(val_ug_g)) + 1
        self.n_items = max(max(train_gi_i), max(train_ui_i)) + 1

        # UG hypergraph
        user_hg_row, user_hg_col = [], []
        group2user_dict = defaultdict(list)
        for u, g in zip(train_ug_u, train_ug_g):
            group2user_dict[g].append(u)

        for g, members in group2user_dict.items():
            user_hg_row.extend(members + [g + self.n_users])
            user_hg_col.extend([g] * (len(members) + 1))

        user_hg = csr_matrix((np.ones(len(user_hg_row)), (np.array(user_hg_row), np.array(user_hg_col))),
                             shape=(self.n_users + self.n_groups, self.n_groups))
        self.user_hg = datautil.convert_sp_mat_to_sp_tensor(datautil.compute_hyper_graph_adj(user_hg))

        user_hg_row, user_hg_col = [], []
        group2user_dict_val = defaultdict(list)
        for u, g in zip(train_ug_u + val_ug_u, train_ug_g + val_ug_g):
            group2user_dict_val[g].append(u)

        for g, members in group2user_dict_val.items():
            user_hg_row.extend(members + [g + self.n_users])
            user_hg_col.extend([g] * (len(members) + 1))

        user_hg_val = csr_matrix((np.ones(len(user_hg_row)), (np.array(user_hg_row), np.array(user_hg_col))),
                                 shape=(self.n_users + self.n_groups, self.n_groups))
        self.user_hg_val = datautil.convert_sp_mat_to_sp_tensor(datautil.compute_hyper_graph_adj(user_hg_val))

        # GI bipartite graph
        gi = csr_matrix((np.ones(len(train_gi_g)), (np.array(train_gi_g), np.array(train_gi_i) + self.n_groups)),
                        shape=(self.n_groups + self.n_items, self.n_groups + self.n_items))
        gi = gi + gi.T
        self.gi_graph = datautil.convert_sp_mat_to_sp_tensor(datautil.compute_normalize_adj(gi))

        # GG weighted graph
        group2item_dict = datautil.load_file_to_dict_format(f"../../../data/{self.dataset}/userItemTrain.txt")
        self.gg_graph_train = torch.FloatTensor(
            datautil.construct_group_graph(group2user_dict, group2item_dict, self.n_groups))
        self.gg_graph_val = torch.FloatTensor(
            datautil.construct_group_graph(group2user_dict_val, group2item_dict, self.n_groups))

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
