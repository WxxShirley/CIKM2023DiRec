from collections import defaultdict
import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import degree
import warnings

warnings.filterwarnings('ignore')


def convert_sp_mat_to_sp_tensor(X):
    """Convert raw sp matrix into sparse tensor"""
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


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


def normalize_edge(edge_index, n_node):
    row, col = edge_index
    deg = degree(col)
    # print(deg)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    return torch.sparse.FloatTensor(edge_index, edge_weight, (n_node, n_node)), edge_weight


def compute_common_interaction_aware_graph(adj_sp, edge_index, num_user, num_item, aug_dict):
    """Referring to Paper CF-GCN(WWW'23), compute user-group-item weighted bipartite graph"""
    # Transform into num_user * num_item shape
    ugi_graph = adj_sp.to_dense()[:num_user, num_user:]

    ugi_graph[ugi_graph > 0] = 1.0
    edge_weight = torch.zeros((num_user + num_item, num_user + num_item))

    for i in range(num_item):
        interacted_ugs = ugi_graph[:, i].nonzero().squeeze(-1)

        items = ugi_graph[interacted_ugs]

        user_user_cap = torch.matmul(items, items.t())
        if aug_dict["method"] == "lhn":
            sim_score = (user_user_cap / ((items.sum(dim=1) * items.sum(dim=1).unsqueeze(-1)))).mean(dim=1)
        else:
            sim_score = (user_user_cap / ((items.sum(dim=1) * items.sum(dim=1).unsqueeze(-1)) ** 0.5)).mean(dim=1)

        edge_weight[interacted_ugs, i + num_user] = sim_score

    for i in range(num_user):
        interacted_items = ugi_graph[i, :].nonzero().squeeze(-1)

        ugs = ugi_graph[:, interacted_items].t()

        item_item_cap = torch.matmul(ugs, ugs.t())
        if aug_dict["method"] == "lhn":
            sim_score = (item_item_cap / ((ugs.sum(dim=1) * ugs.sum(dim=1).unsqueeze(-1)))).mean(dim=1)
        else:
            sim_score = (item_item_cap / ((ugs.sum(dim=1) * ugs.sum(dim=1).unsqueeze(-1)) ** 0.5)).mean(dim=1)

        edge_weight[num_user + interacted_items, i] = sim_score

    edge_weight = edge_weight[edge_index[0], edge_index[1]]
    return edge_weight


def compute_customized_interaction_graph(train_ui_u, train_ui_i, n_user, n_item, aug_dict=None):
    # Step1 create interactive graph
    adj_row = train_ui_u
    adj_col = [i + n_user for i in train_ui_i]

    edge_index = torch.LongTensor([adj_row + adj_col, adj_col + adj_row])

    # Step2 obtain raw interaction weight based on the constructed graph
    adj_sp, edge_weight = normalize_edge(edge_index, n_user + n_item)

    if aug_dict is None or aug_dict["coef"] == 0.0:
        trend = edge_weight
    else:
        # Step3 Reweight
        new_edge_weight = compute_common_interaction_aware_graph(adj_sp, edge_index, n_user, n_item, aug_dict)
        # print(new_edge_weight)
        trend = aug_dict["coef"] * new_edge_weight + edge_weight

    return trend, edge_index


def construct_corrupt_ui_edges(num_user, num_item, user2item_dict, num_edges):
    random_ui_u, random_ui_i = [], []

    for _ in range(num_edges):
        random_u = np.random.randint(num_user)
        random_i = np.random.randint(num_item)

        while random_i in user2item_dict.get(random_u, []):
            random_i = np.random.randint(num_item)

        random_ui_u.append(random_u)
        random_ui_i.append(random_i)

    return random_ui_u, random_ui_i
