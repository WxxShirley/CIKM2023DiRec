from collections import defaultdict
import torch
import numpy as np
import scipy.sparse as sp


def convert_sp_mat_to_sp_tensor(X):
    """Convert raw sp matrix into sparse tensor"""
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


def compute_normalize_adj(adj):
    row_sum = np.array(adj.sum(1))
    d_inv = np.power(row_sum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv)
    return norm_adj


def load_file_to_dict_format(filename):
    """Convert data into Dict format"""
    contents = open(filename, 'r').readlines()
    ui_test_dict = defaultdict(list)
    for content in contents:
        content = content.split()
        u, items = int(content[0]), [int(i) for i in content[1:]]
        ui_test_dict[u] = items
    return ui_test_dict


def load_file_to_list_format(filename):
    """Convert data into two List(s) format"""
    contents = open(filename, 'r').readlines()
    train_u, train_i = [], []
    for content in contents:
        content = content.split()
        u = int(content[0])
        train_u.extend([u] * len(content[1:]))
        for i in content[1:]:
            train_i.append(int(i))

    return train_u, train_i
