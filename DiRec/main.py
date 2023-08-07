import dataloader
import argparse
import numpy as np
import random
import torch
import model
from datetime import datetime
import torch.optim
import time
import metrics
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def set_seed(seed):
    """fix seed"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Hyper paramters
    parser.add_argument("--dataset", type=str, default="Mafengwo")
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--layers", default=2, type=int)

    parser.add_argument("--num_negatives", type=int, default=10)
    parser.add_argument("--reg_coef", type=float, default=1e-4)
    parser.add_argument("--ssl_coef", type=float, default=1e-4)
    parser.add_argument("--user_side_ssl", type=int, default=1)
    parser.add_argument("--group_side_ssl", type=int, default=1)

    # Training parameters
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_times", type=int, default=3)
    parser.add_argument("--run_seeds", type=list, default=[0, 2023, 42, 9999, 12345])
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2048)

    parser.add_argument("--device", type=str, default="cuda:0")
    # parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--topK", type=list, default=[5, 10, 20])

    parser.add_argument("--reweight_coef", type=float, default=0.01)

    args = parser.parse_args()
    device = torch.device(args.device)
    print('= ' * 20)
    print('## Starting Time:', datetime.now().strftime("%Y-%m-%d %H-%M-%S"), flush=True)
    print(args)

    aug_dict = {"hop": 1, "coef": args.reweight_coef, "method": "sc"}
    print(aug_dict)

    dataset = dataloader.Data(dataset_name=args.dataset, batch_size=args.batch_size, num_negatives=args.num_negatives,
                              aug_dict=aug_dict)
    num_user, num_item, num_group = dataset.n_users, dataset.n_items, dataset.n_groups
    print(f"#Users {num_user}, #Groups {num_group}, #Items {num_item} \n")
    hg_dict = {
        "UserHGTrain": dataset.user_hg_train.to(device),
        "UserHGVal": dataset.user_hg_val.to(device),
        "ItemHG": dataset.item_hg.to(device),
        "UITrend": dataset.ui_trend.to(device),
        "UIEdge": dataset.ui_edge.to(device),
    }

    final_recalls, final_ndcgs = [], []
    for t in range(args.run_times):
        # set_seed(args.run_seeds[t])
        # print(f"Setting seed={args.run_seeds[t]}\n")

        rec_model = model.DualIntentRecModel(num_user, num_item, num_group, args.emb_dim, args.layers, hg_dict,
                                             args.user_side_ssl, args.group_side_ssl)
        rec_model.to(device)
        optimizer = torch.optim.Adam(rec_model.parameters(), lr=args.lr)

        best_recalls, best_ndcgs = None, None
        best_recall_0, best_epoch, cnt = 0, 0, 0

        for epoch_id in range(args.epoch):
            train_loader = dataset.get_user_dataloader(batch_size=args.batch_size)

            epoch_loss = 0.0
            ssl_losses = 0.0
            reg_losses = 0.0
            bpr_losses = 0.0
            start_time = time.time()

            for batch_id, (u, pos_i, neg_i) in enumerate(train_loader):
                rec_model.train()
                user_input, pos_item, neg_item = u.to(device), pos_i.to(device), neg_i.to(device)

                bpr_loss, reg_loss, ssl_loss = rec_model.bpr_loss(user_input, pos_item, neg_item)
                if args.user_side_ssl == 0 and args.group_side_ssl == 0:
                    loss = bpr_loss + args.reg_coef * reg_loss
                else:
                    loss = bpr_loss + args.reg_coef * reg_loss + args.ssl_coef * ssl_loss
                    ssl_losses += ssl_loss
                reg_losses += reg_loss
                bpr_losses += bpr_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(
                f"[Epoch {epoch_id + 1}] Train Loss {epoch_loss / len(train_loader):.5f}, BPR Loss {bpr_losses / len(train_loader):.4f}, SSL Loss {ssl_losses / len(train_loader):.5f}, Reg Loss {reg_losses / len(train_loader):.3f}, "
                f"Cost Time {time.time() - start_time:.3f}/s")
            if epoch_id % 5 == 0:
                val_recalls, val_ndcgs = metrics.evaluate(rec_model, dataset, device, args.topK, mode="val")
                test_recalls, test_ndcgs = metrics.evaluate(rec_model, dataset, device, args.topK, mode="test")

                print(f"[Epoch {epoch_id + 1}] Val Recall@{args.topK}: {val_recalls}, Val NDCG@{args.topK} {val_ndcgs}")
                print(f"[Epoch {epoch_id + 1}] Test Recall@{args.topK}: {test_recalls}, Test NDCG@{args.topK} {test_ndcgs} \n")

                if val_ndcgs[-1] > best_recall_0:
                    best_recall_0 = val_ndcgs[-1]

                    # torch.save(rec_model.state_dict(), f"logs/{args.dataset}/model.pt")
                    best_recalls, best_ndcgs = test_recalls, test_ndcgs
                    best_epoch = epoch_id
                    cnt = 0
                else:
                    cnt += 1
                if cnt >= args.patience:
                    break
        # rec_model.load_state_dict(torch.load(f"logs/{args.dataset}/model.pt"))
        # best_recalls, best_ndcgs = metrics.evaluate(rec_model, dataset, device, args.topK, mode="test")
        print(
            f"\nBest Epoch{best_epoch + 1} Recall@{args.topK}: {best_recalls}, Best NDCG@{args.topK}: {best_ndcgs} \n\n")
        final_recalls.append(best_recalls)
        final_ndcgs.append(best_ndcgs)

        del rec_model, optimizer

avg_recall = np.round(np.mean(np.array(final_recalls), axis=0), decimals=4)
std_recall = np.round(np.std(np.array(final_recalls), axis=0), decimals=4)
avg_ndcg = np.round(np.mean(np.array(final_ndcgs), axis=0), decimals=4)
std_ndcg = np.round(np.std(np.array(final_ndcgs), axis=0), decimals=4)
print(final_recalls, final_ndcgs)
print(f"Avg Recall@{args.topK} {avg_recall}+-{std_recall}, Avg NDCG@{args.topK} {avg_ndcg}+-{std_ndcg}")
print('\n## Finishing Time:', datetime.now().strftime("%Y-%m-%d %H-%M-%S"), flush=True)
print('= ' * 20)
print("Done!")
