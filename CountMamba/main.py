import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import argparse
import os
from timm.loss import LabelSmoothingCrossEntropy
import tqdm
import warnings
from torch.utils.data import DataLoader
import torch.nn.functional as F

from util import adjust_learning_rate, evaluate, process_BAPM, process_one_hot
from dataset import CountDataset
from model_CountMamba import CountMambaModel

warnings.filterwarnings("ignore")

# Config
parser = argparse.ArgumentParser(description="WFlib")

# Dataset
parser.add_argument('--dataset', default="CW")
parser.add_argument('--seed', default=2024, type=int,
                    help="Random seed (default: 2024)")
parser.add_argument('--checkpoint-file', default=None, type=str,
                    help="Optional exact checkpoint path; preserves the legacy path when omitted")

# Training Strategy
parser.add_argument('--early_stage', action="store_true")
parser.add_argument('--num_aug', default=20, type=int)

parser.add_argument('--num_tabs', default=1, type=int)
parser.add_argument('--fine_predict', action="store_true")

# Representation
parser.add_argument('--seq_len', default=5000, type=int)
parser.add_argument('--maximum_load_time', default=80, type=int)
parser.add_argument('--max_matrix_len', default=1800, type=int)
parser.add_argument('--time_interval_threshold', default=0.1, type=float)
parser.add_argument('--maximum_cell_number', default=2, type=int)
parser.add_argument('--log_transform', action="store_true")

# Model
parser.add_argument('--drop_path_rate', default=0.2, type=float)
parser.add_argument('--depth', default=3, type=int)
parser.add_argument('--embed_dim', default=256, type=int)
# Train Parameter
parser.add_argument('--batch_size', default=200, type=int)
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--min_lr', type=float, default=1e-6)
parser.add_argument('--layer_decay', type=float, default=0.75)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--warmup_epochs', type=int, default=0)

args = parser.parse_args()

# Set a fixed seed for reproducibility.  Keep the historical default while
# allowing the experiment runner to select a seed per training job.
fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.cuda.manual_seed(fix_seed)
np.random.seed(fix_seed)
rng = np.random.RandomState(fix_seed)
cudnn.benchmark = False
cudnn.deterministic = True

ckp_path = os.path.join("../checkpoints/", args.dataset, "CountMamba")
os.makedirs(ckp_path, exist_ok=True)
out_file = args.checkpoint_file or os.path.join(ckp_path, f"max_f1.pth")
os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)

print(f"--------------------------------\n"
      f"Windowed Traffic Counting Matrix\n"
      f"Sequence Length: {args.seq_len}\n"
      f"Maximum Load Time: {args.maximum_load_time}s\n"
      f"Number of Columns in Matrix: {args.max_matrix_len}\n"
      f"Time Interval Threshold: {args.time_interval_threshold}\n"
      f"Maximum Cell Number: {args.maximum_cell_number}\n"
      f"Use Log Transformation: {args.log_transform}\n"
      f"=>Time Window Length: {int(args.maximum_load_time / args.max_matrix_len * 1000)}ms\n"
      f"--------------------------------")


# Dataset
def load_data(data_path):
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]

    if args.fine_predict:
        BAPM = data["BAPM"]
        return X, BAPM, y
    else:
        return X, y


if args.fine_predict:
    train_X, train_BAPM, train_y = load_data(os.path.join("../npz_dataset", args.dataset, f"train.npz"))
    valid_X, valid_BAPM, valid_y = load_data(os.path.join("../npz_dataset", args.dataset, f"valid.npz"))
else:
    train_X, train_y = load_data(os.path.join("../npz_dataset", args.dataset, f"train.npz"))
    valid_X, valid_y = load_data(os.path.join("../npz_dataset", args.dataset, f"valid.npz"))
if args.num_tabs == 1:
    num_classes = len(np.unique(train_y))
else:
    num_classes = train_y.shape[1]

print(f"Dataset: {args.dataset}\n"
      f"Train: {len(train_X)}, Val: {len(valid_X)}\n"
      f"--------------------------------")

if args.fine_predict:
    dataset_train = CountDataset(train_X, train_y, args=args, BAPM=train_BAPM)
    dataset_val = CountDataset(valid_X, valid_y, args=args, BAPM=valid_BAPM)
else:
    dataset_train = CountDataset(train_X, train_y, args=args)
    dataset_val = CountDataset(valid_X, valid_y, args=args)

data_loader_train = DataLoader(
    dataset_train,
    shuffle=True,
    batch_size=args.batch_size,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

data_loader_val = DataLoader(
    dataset_val,
    shuffle=False,
    batch_size=args.batch_size,
    num_workers=4,
    pin_memory=True,
    drop_last=False
)

# Model
print(f"CountMamba\n"
      f"Number of Augmentation: {args.num_aug}\n"
      f"Depth: {args.depth}\n"
      f"Drop Path Rate: {args.drop_path_rate}\n"
      f"Embed Dimension: {args.embed_dim}\n"
      f"--------------------------------")

patch_size = 2 * (args.maximum_cell_number + 1) + 2
model = CountMambaModel(num_classes=num_classes, drop_path_rate=args.drop_path_rate, depth=args.depth,
                        embed_dim=args.embed_dim, patch_size=patch_size, max_matrix_len=args.max_matrix_len,
                        early_stage=args.early_stage, num_tabs=args.num_tabs, fine_predict=args.fine_predict)
model.cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

if args.num_tabs > 1:
    criterion = torch.nn.MultiLabelSoftMarginLoss()
else:
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

# Train
metric_best_value = 0
for epoch in range(args.epochs):
    model.train()
    sum_loss = 0
    sum_count = 0
    for index, cur_data in enumerate(tqdm.tqdm(data_loader_train)):
        adjust_learning_rate(optimizer, index / len(data_loader_train) + epoch, args)

        cur_X, cur_y = cur_data[0][0].cuda(), cur_data[1].cuda().long()
        idx = cur_data[0][1].cuda()
        optimizer.zero_grad()

        if not args.early_stage and not args.fine_predict:
            outs = model(cur_X, idx)
            loss = criterion(outs, cur_y)
        elif args.fine_predict:
            # fine-predict training
            BAPM = cur_data[0][2]
            BAPM = process_BAPM(BAPM, num_classes - 1)

            outs, fine_out, sliding_size = model(cur_X, idx)
            loss = criterion(outs, cur_y)

            BAPM = torch.roll(BAPM, shifts=sliding_size, dims=-1)
            one_hot = process_one_hot(BAPM, num_classes - 1, model.num_patches)
            one_hot = one_hot.cuda()

            loss_fine = criterion(fine_out.view(-1, num_classes + 1), one_hot.view(-1, num_classes + 1))
            loss += loss_fine
        else:
            # early-stage training
            outs = model(cur_X, idx)

            N, L, D = outs.shape
            noise = torch.rand(N, L, device=outs.device)
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_keep = ids_shuffle[:, :args.num_aug]
            outs_masked = torch.gather(outs, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            cur_y = cur_y.unsqueeze(1).repeat(1, args.num_aug)

            loss = criterion(outs_masked.contiguous().view(-1, D), cur_y.contiguous().view(-1))

        loss.backward()
        optimizer.step()

        sum_loss += loss.data.cpu().numpy() * outs.shape[0]
        sum_count += outs.shape[0]

    train_loss = round(sum_loss / sum_count, 3)
    print(f"epoch {epoch}: train_loss = {train_loss}")

    valid_result = evaluate(model, data_loader_val, args.num_tabs, num_classes, args.fine_predict)
    print(f"At Epoch {epoch}")
    print(f"Result on Val Dataset: {valid_result}")
    print(f"lr: {optimizer.param_groups[0]['lr']}")

    if args.num_tabs > 1:
        if valid_result[f"ap@{args.num_tabs}"] > metric_best_value:
            metric_best_value = valid_result[f"ap@{args.num_tabs}"]
            torch.save(model.state_dict(), out_file)
    else:
        if valid_result["F1-score"] > metric_best_value:
            metric_best_value = valid_result["F1-score"]
            torch.save(model.state_dict(), out_file)
