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


def compute_hyper_graph_adj(h):
    """Based on Hyper-graph matrix H, compute final normalized adjacency matrix"""
    dv = np.array(h.sum(1))
    d_inv = np.power(dv, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    dv = sp.diags(d_inv)

    de = np.array(h.sum(0))
    d_ine = np.power(de, -1).flatten()
    d_ine[np.isinf(d_ine)] = 0.
    de = sp.diags(d_ine)

    # D_v^{-\frac{1}{2}} H D_e^{-1} H^T D_v^{-\frac{1}{2}}
    return dv.dot(h).dot(de).dot(h.T).dot(dv)


def construct_group_graph(gu_dict, gi_dict, num_groups):
    matrix = np.zeros((num_groups, num_groups))

    for g in range(num_groups):
        g1_member = set(gu_dict.get(g, []))
        g1_item = set(gi_dict.get(g, []))

        for g2 in range(g + 1, num_groups):
            g2_member = set(gu_dict.get(g2, []))
            g2_item = set(gi_dict.get(g2, []))

            member_overlap = g1_member & g2_member
            member_union = g1_member | g2_member

            item_overlap = g1_item & g2_item
            item_union = g1_item | g2_item

            if (len(member_union) + len(item_union)) == 0:
                matrix[g][g2] = matrix[g2][g] = 0.0
                continue

            matrix[g][g2] = float((len(member_overlap) + len(item_overlap)) / (len(member_union) + len(item_union)))
            matrix[g2][g] = matrix[g][g2]

    matrix = matrix + np.diag([1.0] * num_groups)
    degree = np.sum(np.array(matrix), 1)
    return np.dot(np.diag(1.0 / degree), matrix)
