import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import argparse
import os
import json
import warnings
from torch.utils.data import DataLoader

from util import evaluate
from dataset import CountDataset
from model_CountMamba import CountMambaModel

warnings.filterwarnings("ignore")

# Config
parser = argparse.ArgumentParser(description="WFlib")

# Dataset
parser.add_argument('--dataset', default="CW")
parser.add_argument('--seed', default=2024, type=int,
                    help="Random seed (default: 2024)")
parser.add_argument("--load_ratio", type=int, default=100)
parser.add_argument("--result_file", type=str, default="test_p100", help="File to save test results")
parser.add_argument('--checkpoint-file', default=None, type=str,
                    help="Optional exact checkpoint path; preserves the legacy path when omitted")
parser.add_argument('--output-file', default=None, type=str,
                    help="Optional exact JSON output path; preserves the legacy path when omitted")
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
# Train
parser.add_argument('--batch_size', default=200, type=int)

args = parser.parse_args()

# Set a fixed seed for reproducibility while retaining the historical default.
fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.cuda.manual_seed(fix_seed)
np.random.seed(fix_seed)
rng = np.random.RandomState(fix_seed)
cudnn.benchmark = False
cudnn.deterministic = True

in_path = os.path.join("../npz_dataset", args.dataset)
log_path = os.path.join("../logs", args.dataset, "CountMamba")
ckp_path = os.path.join("../checkpoints/", args.dataset, "CountMamba")
os.makedirs(log_path, exist_ok=True)
out_file = args.output_file or os.path.join(log_path, f"{args.result_file}.json")
os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)


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


if args.load_ratio != 100:
    test_X, test_y = load_data(os.path.join(in_path, f"test_p{args.load_ratio}.npz"))
else:
    if args.fine_predict:
        test_X, test_BAPM, test_y = load_data(os.path.join(in_path, f"test.npz"))
    else:
        test_X, test_y = load_data(os.path.join(in_path, f"test.npz"))

if args.num_tabs == 1:
    num_classes = len(np.unique(test_y))
else:
    num_classes = test_y.shape[1]

if args.fine_predict:
    dataset_test = CountDataset(test_X, test_y, args=args, BAPM=test_BAPM)
else:
    dataset_test = CountDataset(test_X, test_y, args=args)
    
data_loader_test = DataLoader(
    dataset_test,
    shuffle=False,
    batch_size=args.batch_size,
    num_workers=4,
    pin_memory=True,
    drop_last=False
)

# Model
patch_size = 2 * (args.maximum_cell_number + 1) + 2
model = CountMambaModel(num_classes=num_classes, drop_path_rate=args.drop_path_rate, depth=args.depth,
                        embed_dim=args.embed_dim, patch_size=patch_size, max_matrix_len=args.max_matrix_len,
                        early_stage=False, num_tabs=args.num_tabs, fine_predict=args.fine_predict)
checkpoint_file = args.checkpoint_file or os.path.join(ckp_path, f"max_f1.pth")
model.load_state_dict(torch.load(checkpoint_file, map_location="cpu"))
model.cuda()

with torch.no_grad():
    model.eval()
    result = evaluate(model, data_loader_test, args.num_tabs, num_classes, args.fine_predict)
    print(result)

    with open(out_file, "w") as fp:
        json.dump(result, fp, indent=4)
