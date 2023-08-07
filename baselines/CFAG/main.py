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
    """Fix seed"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="Steam")
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--contexual_emb_dim", type=int, default=256)
    parser.add_argument("--layers", default=1, type=int)

    parser.add_argument("--num_negatives", type=int, default=10)
    parser.add_argument("--reg_coef", type=float, default=1e-2)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--run_seeds", type=list, default=[0, 2023, 42])
    parser.add_argument("--run_times", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--topK", type=list, default=[5, 10, 20])
    parser.add_argument("--attn_coef", type=float, default=0.1)

    args = parser.parse_args()
    device = torch.device(args.device)
    print('= ' * 20)
    print('## Starting Time:', datetime.now().strftime("%Y-%m-%d %H-%M-%S"), flush=True)
    print(args)

    dataset = dataloader.Data(dataset_name=args.dataset, batch_size=args.batch_size, num_negatives=args.num_negatives)
    norm_ug, norm_ug_val, norm_ui, norm_gi, r, r_val, r_ui = dataset.prepare_data()
    num_user, num_group, num_item = dataset.n_users, dataset.n_groups, dataset.n_items
    # print(f"Main #Users {num_user}, #Items {num_item}, #Groups {num_group}")
    norm_ug, norm_ui, norm_gi, r, r_ui = norm_ug.to(device), norm_ui.to(device), norm_gi.to(device), r.to(
        device), r_ui.to(device)
    norm_ug_val, r_val = norm_ug_val.to(device), r_val.to(device)
    # print(r)

    avg_recalls, avg_ndcgs = [], []

    for t in range(args.run_times):
        # seed = args.run_seeds[t]
        # set_seed(seed)
        # print(f"Seed {seed} \n",)

        rec_model = model.CFGA(num_user, num_item, num_group, args.layers, args.emb_dim, norm_ug, norm_ug_val,
                               norm_adj_gi=norm_gi, norm_adj_ui=norm_ui, R=r, R_val=r_val, R_ui=r_ui,
                               att_coef=args.attn_coef, contexual_emb_dim=args.contexual_emb_dim)
        rec_model.to(device)
        optimizer = torch.optim.Adam(rec_model.parameters(), lr=args.lr)

        best_recalls, best_ndcgs = None, None
        best_recall_0, best_epoch = 0, 0

        for epoch_id in range(args.epoch):
            train_loader = dataset.get_user_dataloader(batch_size=args.batch_size)

            epoch_loss = 0.0
            start_time = time.time()

            for batch_id, (u, pos_i, neg_i) in enumerate(train_loader):
                rec_model.train()
                user_input, pos_item, neg_item = u.to(device), pos_i.to(device), neg_i.to(device)

                bpr_loss, reg_loss = rec_model.bpr_loss(user_input, pos_item, neg_item)
                loss = bpr_loss + args.reg_coef * reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"[Epoch {epoch_id + 1}] Train Loss {epoch_loss / len(train_loader):.4f}, "
                  f"Cost Time {time.time() - start_time:.3f}/s")

            if epoch_id % 5 == 0:
                val_recalls, val_ndcgs = metrics.evaluate(rec_model, dataset, device, args.topK, mode="val")
                test_recalls, test_ndcgs = metrics.evaluate(rec_model, dataset, device, args.topK, mode="test")
                if val_ndcgs[-1] > best_recall_0:
                    best_recall_0 = val_ndcgs[-1]
                    best_recalls, best_ndcgs = test_recalls, test_ndcgs
                    best_epoch = epoch_id
                    cnt = 0
                else:
                    cnt += 1
                print(f"[Epoch {epoch_id + 1}] Val Recall@{args.topK}: {val_recalls}, Val NDCG@{args.topK} {val_ndcgs}")
                print(
                    f"[Epoch {epoch_id + 1}] Test Recall@{args.topK}: {test_recalls}, Test NDCG@{args.topK} {test_ndcgs} \n")

                if cnt >= 10:
                    break

        avg_recalls.append(best_recalls)
        avg_ndcgs.append(best_ndcgs)
        print(
            f"\nBest Epoch{best_epoch + 1} Recall@{args.topK}: {best_recalls}, Best NDCG@{args.topK}: {best_ndcgs} \n\n")

std_recall = np.round(np.std(np.array(avg_recalls), axis=0), decimals=4)
avg_recall = np.round(np.mean(np.array(avg_recalls), axis=0), decimals=4)
std_ndcg = np.round(np.std(np.array(avg_ndcgs), axis=0), decimals=4)
avg_ndcg = np.round(np.mean(np.array(avg_ndcgs), axis=0), decimals=4)
print(f"Avg Recall@{args.topK} {avg_recall}+/-{std_recall}, Avg NDCG@{args.topK} {avg_ndcg}+/-{std_ndcg}")
print('\n## Finishing Time:', datetime.now().strftime("%Y-%m-%d %H-%M-%S"), flush=True)
print('= ' * 20)
print("Done!")
