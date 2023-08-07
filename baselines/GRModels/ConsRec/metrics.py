import numpy as np
import copy
import model
import dataloader


def recall(rank, ground_truth):
    hits = [1.0 if item in ground_truth else 0.0 for item in rank]
    return np.sum(hits) / len(ground_truth)


def ndcg(rank, ground_truth):
    idcg_len = min(len(rank), len(ground_truth))

    idcg = (1.0 / np.log2(np.arange(2, idcg_len + 2)))
    dcg = np.sum([1.0 / np.log2(idx + 2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
    return dcg / np.sum(idcg)


def evaluate(rec_model: model.ConsRec4RGI, dataset: dataloader.Data, device, k_list=[10, 20, 50], mode="val"):
    rec_model.eval()
    # [num_user, num_item]
    all_ratings = rec_model.get_user_rating(mode=mode).detach().cpu().numpy()

    all_recall, all_ndcg = [], []

    ug_dict = dataset.ug_val_dict if mode == "val" else dataset.ug_test_dict

    for k in k_list:
        ratings = copy.deepcopy(all_ratings)
        recalls, ndcgs = [], []
        for user, truth_groups in ug_dict.items():
            if len(truth_groups) == 0:
                print(mode, user)
                continue
            pred_score = copy.deepcopy(ratings[user, :])
            # print(pred_score)
            observed_items = dataset.ug_train_dict[user] if mode == "val" else dataset.ug_train_dict[
                                                                                   user] + dataset.ug_val_dict.get(user,
                                                                                                                   [])
            pred_score[observed_items] = -np.inf
            pred_rank = np.argsort(-pred_score)[:k]
            # print(pred_rank)
            # if user % 1000 == 0 and k == 5:
            #     random_idx = np.random.choice(dataset.n_groups, 5)
            #     print(f"测试用户{user}, 真值群组{dataset.ug_test_dict[user]}, 预测前10{pred_rank}, 对应得分{pred_score[pred_rank]}, 随机得分{pred_score[random_idx]}")

            recalls.append(recall(pred_rank, truth_groups))
            ndcgs.append(ndcg(pred_rank, truth_groups))

            if mode == "test":
                if np.isnan(recalls[-1]):
                    print(user, pred_rank, truth_groups)

        cur_recall, cur_ndcg = np.round(np.mean(recalls), decimals=4), np.round(np.mean(ndcgs), decimals=4)
        all_recall.append(cur_recall)
        all_ndcg.append(cur_ndcg)

    return all_recall, all_ndcg
