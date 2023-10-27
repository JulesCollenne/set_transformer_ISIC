import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from data_isic import ModelFetcherISIC, DataLoaderISIC
from data_modelnet40 import ModelFetcher
from models import SetTransformer
from modules import ISAB, PMA, SAB


# class SetTransformer(nn.Module):
#     def __init__(
#         self,
#         dim_input=100,
#         num_outputs=1,
#         dim_output=2,
#         num_inds=32,
#         dim_hidden=128,
#         num_heads=4,
#         ln=False,
#     ):
#         super(SetTransformer, self).__init__()
#         self.enc = nn.Sequential(
#             ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
#             ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
#         )
#         self.dec = nn.Sequential(
#             nn.Dropout(),
#             PMA(dim_hidden, num_heads, num_outputs, ln=ln),
#             nn.Dropout(),
#             nn.Linear(dim_hidden, dim_output),
#         )
#
#     def forward(self, X):
#         return self.dec(self.enc(X)).squeeze()


parser = argparse.ArgumentParser()
# parser.add_argument("--num_pts", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dim", type=int, default=256)
parser.add_argument("--n_heads", type=int, default=4)
parser.add_argument("--n_anc", type=int, default=16)
parser.add_argument("--train_epochs", type=int, default=2000)
args = parser.parse_args()
args.exp_name = f"Nd{args.dim}h{args.n_heads}i{args.n_anc}_lr{args.learning_rate}bs{args.batch_size}"
log_dir = "result/" + args.exp_name
model_path = log_dir + "/model"
writer = SummaryWriter(log_dir)

# generator = ModelFetcherISIC(
#     "features/simsiam_train.csv",
#     "features/simsiam_val.csv",
#     "GroundTruth.csv",
#     args.batch_size,
#     do_standardize=False,
#     do_augmentation=False,
# )


train_gen = DataLoaderISIC(
    "features/simsiam_train.csv",
    "GroundTruth.csv",
    batch_size=args.batch_size,
)

val_gen = DataLoaderISIC(
    "features/simsiam_val.csv",
    "GroundTruth.csv",
    batch_size=args.batch_size,
)

test_gen = DataLoaderISIC(
    "features/simsiam_test.csv",
    "GroundTruth.csv",
    batch_size=args.batch_size,
)

# model = SetTransformer(dim_hidden=args.dim, num_heads=args.n_heads, num_inds=args.n_anc)
model = SetTransformer(100, 20, 20, num_inds=20)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()
model = nn.DataParallel(model)
model = model.cuda()

for epoch in range(args.train_epochs):
    model.train()
    losses, total, correct = [], 0, 0
    for imgs, lbls in train_gen.train_data():
        imgs = torch.Tensor(imgs).cuda()
        lbls = torch.Tensor(lbls).long().cuda()
        preds = model(imgs)
        loss = criterion(preds, lbls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        total += lbls.shape[0]
        correct += (preds.argmax(dim=1) == lbls).sum().item()

    avg_loss, avg_acc = np.mean(losses), correct / total
    writer.add_scalar("train_loss", avg_loss)
    writer.add_scalar("train_acc", avg_acc)
    print(f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f}")

    if epoch % 10 == 0:
        model.eval()
        losses, total, correct = [], 0, 0
        for imgs, lbls in val_gen.train_data():
            imgs = torch.Tensor(imgs).cuda()
            lbls = torch.Tensor(lbls).long().cuda()
            preds = model(imgs)
            loss = criterion(preds, lbls)

            losses.append(loss.item())
            total += lbls.shape[0]
            correct += (preds.argmax(dim=1) == lbls).sum().item()
        avg_loss, avg_acc = np.mean(losses), correct / total
        writer.add_scalar("test_loss", avg_loss)
        writer.add_scalar("test_acc", avg_acc)
        print(f"Epoch {epoch}: test loss {avg_loss:.3f} test acc {avg_acc:.3f}")
