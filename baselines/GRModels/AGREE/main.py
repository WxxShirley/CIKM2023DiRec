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

    # [Mafengwo, Steam, Weeplaces]
    parser.add_argument("--dataset", type=str, default="Mafengwo")

    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--num_negatives", type=int, default=10)

    # 训练参数
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--run_times", type=int, default=5)
    parser.add_argument("--run_seeds", type=list, default=[0, 2023, 42, 12345, 9999])
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2048)

    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--topK", type=list, default=[5, 10, 20])

    args = parser.parse_args()
    device = torch.device(args.device)
    print('= ' * 20)
    print('## Starting Time:', datetime.now().strftime("%Y-%m-%d %H-%M-%S"), flush=True)
    print(args)

    dataset = dataloader.Data(dataset_name=args.dataset, batch_size=args.batch_size, num_negatives=args.num_negatives)
    num_user, num_group = dataset.n_users, dataset.n_groups
    print(f"#Users {num_user}, #Groups {num_group} \n")

    train_group_users, train_group_masks = dataset.train_group_users.to(device), dataset.train_group_masks.to(device)
    val_group_users, val_group_masks = dataset.val_group_users.to(device), dataset.val_group_masks.to(device)

    final_recalls, final_ndcgs = [], []
    for t in range(args.run_times):
        set_seed(args.run_seeds[t])
        print(f"Seed {args.run_seeds[t]} ...  \n")
        rec_model = model.AGREE4Rec(num_user, num_group, args.emb_dim, 0.1, train_group_users, train_group_masks, val_group_users, val_group_masks)

        rec_model.to(device)
        optimizer = torch.optim.Adam(rec_model.parameters(), lr=args.lr)

        best_recalls, best_ndcgs = None, None
        best_recall_0, best_epoch, cnt = 0, 0, 0

        for epoch_id in range(args.epoch):
            rec_model.train()

            train_loadere = dataset.get_user_dataloader(batch_size=args.batch_size)

            epoch_loss = 0.0

            start_time = time.time()

            for batch_id, (u, pos_i, neg_i) in enumerate(train_loadere):
                user_input, pos_input, neg_input = u.to(device), pos_i.to(device), neg_i.to(device)

                pos_group_user, pos_group_mask = train_group_users[pos_input], train_group_masks[pos_input]
                neg_group_user, neg_group_mask = train_group_users[neg_input], train_group_masks[neg_input]

                bpr_loss = rec_model(user_input, pos_group_user, pos_group_mask, neg_group_user, neg_group_mask)
                loss = bpr_loss

                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(
                f"[Epoch {epoch_id + 1}] Train Loss {epoch_loss / len(train_loadere):.5f}, Cost Time {time.time() - start_time:.3f}/s")
            if epoch_id % 5 == 0:
                val_recalls, val_ndcgs = metrics.evaluate(rec_model, dataset, device, args.topK, mode="val")
                test_recalls, test_ndcgs = metrics.evaluate(rec_model, dataset, device, args.topK, mode="test")

                print(f"[Epoch {epoch_id + 1}] Val Recall@{args.topK}: {val_recalls}, Val NDCG@{args.topK} {val_ndcgs}")
                print(
                    f"[Epoch {epoch_id + 1}] Test Recall@{args.topK}: {test_recalls}, Test NDCG@{args.topK} {test_ndcgs} \n")

                if val_ndcgs[-1] > best_recall_0:
                    best_recall_0 = val_ndcgs[-1]
                    best_recalls, best_ndcgs = test_recalls, test_ndcgs
                    best_epoch = epoch_id
                    cnt = 0
                else:
                    cnt += 1
                if cnt >= args.patience:
                    break
        print(
            f"\nBest Epoch{best_epoch + 1} Recall@{args.topK}: {best_recalls}, Best NDCG@{args.topK}: {best_ndcgs} \n\n")
        final_recalls.append(best_recalls)
        final_ndcgs.append(best_ndcgs)


avg_recall = np.round(np.mean(np.array(final_recalls), axis=0), decimals=4)
std_recall = np.round(np.std(np.array(final_recalls), axis=0), decimals=4)
avg_ndcg = np.round(np.mean(np.array(final_ndcgs), axis=0), decimals=4)
std_ndcg = np.round(np.std(np.array(final_ndcgs), axis=0), decimals=4)
print(f"Avg Recall@{args.topK} {avg_recall}+/-{std_recall}, Avg NDCG@{args.topK} {avg_ndcg}+/-{std_ndcg}")
print('\n## Finishing Time:', datetime.now().strftime("%Y-%m-%d %H-%M-%S"), flush=True)
print('= ' * 20)
print("Done!")
