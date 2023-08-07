import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from torch.utils.data import TensorDataset, DataLoader
import datautil
import torch


class Data(object):
    def __init__(self, dataset_name="Mafengwo", batch_size=1024, num_negatives=4):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_negative = num_negatives

        self.ug_train_dict = datautil.load_file_to_dict_format(f"../data/{dataset_name}/train.txt")
        self.ug_test_dict = datautil.load_file_to_dict_format(f"../data/{dataset_name}/test.txt")
        self.ug_val_dict = datautil.load_file_to_dict_format(f"../data/{dataset_name}/val.txt")

        self.n_groups, self.n_users, self.n_items = 0, 0, 0
        self.n_train, self.n_test, self.n_val = 0, 0, 0

    def prepare_data(self):
        train_ui_u, train_ui_i = datautil.load_file_to_list_format(f"../data/{self.dataset_name}/groupItemTrain.txt")
        train_gi_g, train_gi_i = datautil.load_file_to_list_format(f"../data/{self.dataset_name}/userItemTrain.txt")
        train_ug_u, train_ug_g = datautil.load_file_to_list_format(f"../data/{self.dataset_name}/train.txt")
        test_ug_u, _ = datautil.load_file_to_list_format(f"../data/{self.dataset_name}/test.txt")
        val_ug_u, val_ug_g = datautil.load_file_to_list_format(f"../data/{self.dataset_name}/val.txt")

        self.n_items = max(max(train_ui_i), max(train_gi_i)) + 1
        self.n_users = max(max(train_ui_u), max(train_ug_u)) + 1
        self.n_groups = max(max(train_gi_g), max(train_ug_g)) + 1

        self.n_train, self.n_test, self.n_val = len(train_ug_u), len(test_ug_u), len(val_ug_u)
        self.print_statistics()

        ug_r = csr_matrix((np.ones(len(train_ug_u)), (np.array(train_ug_u), np.array(train_ug_g))),
                          shape=(self.n_users, self.n_groups))
        ug_r = datautil.convert_sp_mat_to_sp_tensor(ug_r)

        # 在测试集上测试使用
        ug_r_val = csr_matrix((np.ones(len(train_ug_u) + len(val_ug_u)),
                               (np.array(train_ug_u + val_ug_u), np.array(train_ug_g + val_ug_g))),
                              shape=(self.n_users, self.n_groups))
        ug_r_val = datautil.convert_sp_mat_to_sp_tensor(ug_r_val)

        ui_r = csr_matrix((np.ones(len(train_ui_u)), (np.array(train_ui_u), np.array(train_ui_i))),
                          shape=(self.n_users, self.n_items))
        ui_r = datautil.convert_sp_mat_to_sp_tensor(ui_r)

        ug_graph = csr_matrix((np.ones(len(train_ug_u)), (np.array(train_ug_u), np.array(train_ug_g) + self.n_users)),
                              shape=(self.n_users + self.n_groups, self.n_users + self.n_groups))
        ug_graph = ug_graph + ug_graph.T
        norm_ug = datautil.compute_normalize_adj(ug_graph + sp.eye(ug_graph.shape[0]))
        # mean_ug = datautil.compute_normalize_adj(ug_graph)

        ug_graph_val = csr_matrix((np.ones(len(train_ug_u) + len(val_ug_u)),
                                   (np.array(train_ug_u + val_ug_u), np.array(train_ug_g + val_ug_g) + self.n_users)),
                                  shape=(self.n_users + self.n_groups, self.n_users + self.n_groups))
        ug_graph_val = ug_graph_val + ug_graph_val.T
        norm_ug_val = datautil.compute_normalize_adj(ug_graph_val + sp.eye(ug_graph_val.shape[0]))

        ui_graph = csr_matrix((np.ones(len(train_ui_u)), (np.array(train_ui_u), np.array(train_ui_i) + self.n_users)),
                              shape=(self.n_users + self.n_items, self.n_users + self.n_items))
        ui_graph = ui_graph + ui_graph.T
        norm_ui = datautil.compute_normalize_adj(ui_graph + sp.eye(ui_graph.shape[0]))
        # mean_ui = datautil.compute_normalize_adj(ui_graph)

        gi_graph = csr_matrix((np.ones(len(train_gi_g)), (np.array(train_gi_g), np.array(train_gi_i) + self.n_groups)),
                              shape=(self.n_groups + self.n_items, self.n_groups + self.n_items))
        gi_graph = gi_graph + gi_graph.T
        norm_gi = datautil.compute_normalize_adj(gi_graph + sp.eye(gi_graph.shape[0]))
        # mean_gi = datautil.compute_normalize_adj(gi_graph)

        return datautil.convert_sp_mat_to_sp_tensor(norm_ug), datautil.convert_sp_mat_to_sp_tensor(
            norm_ug_val), datautil.convert_sp_mat_to_sp_tensor(norm_ui), \
               datautil.convert_sp_mat_to_sp_tensor(norm_gi), ug_r, ug_r_val, ui_r

    def print_statistics(self):
        print(f"#Groups {self.n_groups}, #Users {self.n_users}, #Items {self.n_items}")
        print(f"#Interactions {self.n_train + self.n_test + self.n_val}")
        print(
            f"#Train interactions {self.n_train}, #Test interactions {self.n_test}, #Val interactions {self.n_val}, Sparsity"
            f" {((self.n_train + self.n_test + self.n_val) / self.n_users / self.n_groups):.5f}")

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
