import numpy as np
from scipy.sparse import csr_matrix
from torch.utils.data import TensorDataset, DataLoader
import datautil
import torch


class Data(object):
    def __init__(self, dataset_name="Mafengwo", batch_size=1024, num_negatives=6):
        print(f"Loading [{dataset_name.upper()}] dataset ... ")

        self.dataset = dataset_name
        self.num_negatives = num_negatives
        self.batch_size = batch_size

        self.ug_train_dict = datautil.load_file_to_dict_format(f"../../data/{dataset_name}/train.txt")
        self.ug_test_dict = datautil.load_file_to_dict_format(f"../../data/{dataset_name}/test.txt")
        self.ug_val_dict = datautil.load_file_to_dict_format(f"../../data/{dataset_name}/val.txt")

        self.n_users, self.n_groups = 0, 0
        self.user_np, self.group_np = None, None
        self.graph, self.graph_val = None, None

        self.prepare_data()

    def prepare_data(self):
        train_ug_u, train_ug_g = datautil.load_file_to_list_format(f"../../data/{self.dataset}/train.txt")
        val_ug_u, val_ug_g = datautil.load_file_to_list_format(f"../../data/{self.dataset}/val.txt")
        self.user_np = train_ug_u
        self.group_np = train_ug_g

        self.n_users = max(max(train_ug_u), max(val_ug_u)) + 1
        self.n_groups = max(max(train_ug_g), max(val_ug_g)) + 1

        user_graph = csr_matrix((np.ones(len(train_ug_u)), (np.array(train_ug_u), np.array(train_ug_g) + self.n_users)),
                                shape=(self.n_users + self.n_groups, self.n_users + self.n_groups))
        user_graph = user_graph + user_graph.T
        self.graph = datautil.convert_sp_mat_to_sp_tensor(datautil.compute_normalize_adj(user_graph))

        user_graph_val = csr_matrix((np.ones(len(train_ug_u) + len(val_ug_u)),
                                     (np.array(train_ug_u + val_ug_u), np.array(train_ug_g + val_ug_g) + self.n_users)),
                                    shape=(self.n_users + self.n_groups, self.n_users + self.n_groups))
        user_graph_val = user_graph_val + user_graph_val.T
        self.graph_val = datautil.convert_sp_mat_to_sp_tensor(datautil.compute_normalize_adj(user_graph_val))

        # print(self.graph)
        # print(self.graph_val)

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
