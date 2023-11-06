import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from torch.utils.tensorboard import SummaryWriter

from data_isic import DataLoaderISIC
from models import SetTransformer

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
parser.add_argument("--train_epochs", type=int, default=200)
args = parser.parse_args()

#models = ("BYOL", "Moco", "SimCLR", "SimSiam")
# models = ("Moco", "SimCLR", "SimSiam")

models = ("BYOL", "Moco")

for model_name in models:
    print("Training:", model_name)
    args.exp_name = f"{model_name}Nd{args.dim}h{args.n_heads}i{args.n_anc}_lr{args.learning_rate}bs{args.batch_size}"
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

    num_inds = 10

    train_gen = DataLoaderISIC(
        f"features/{model_name}_train.csv",
        "GroundTruth.csv",
        batch_size=args.batch_size,
        input_dim=num_inds
    )

    val_gen = DataLoaderISIC(
        f"features/{model_name}_val.csv",
        "GroundTruth.csv",
        batch_size=args.batch_size,
        input_dim=num_inds
    )

    test_gen = DataLoaderISIC(
        f"features/{model_name}_test.csv",
        "GroundTruth.csv",
        batch_size=args.batch_size,
        input_dim=num_inds
    )

    # model = SetTransformer(dim_hidden=args.dim, num_heads=args.n_heads, num_inds=args.n_anc)
    n_features = sum(['features' in col for col in pd.read_csv(f"features/{model_name}_val.csv").columns])
    model = SetTransformer(n_features, 10, 2, num_inds=num_inds)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    class_weights = torch.tensor([0.02, 0.98]).cuda()
    criterion = nn.CrossEntropyLoss(class_weights)
    model = nn.DataParallel(model)
    model = model.cuda()
    old_mean = 0

    for epoch in range(args.train_epochs):
        model.train()
        losses, total, correct, true_labels, predicted_probs = [], 0, 0, [], []

        for imgs, lbls in train_gen.train_data():
            print(imgs.shape)
            print(lbls.shape)
            exit()

            imgs = torch.Tensor(imgs).cuda()
            lbls = torch.Tensor(lbls).long().cuda()
            preds = model(imgs)

            # loss = criterion(preds, lbls)
            loss = criterion(preds.view(-1, 2), lbls.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            total += lbls.view(-1).shape[0]
            correct += (preds.view(-1, 2).argmax(dim=1) == lbls.view(-1)).sum().item()

            true_labels += lbls.view(-1).cpu().numpy().tolist()
            predicted_probs += torch.softmax(preds.view(-1, 2), dim=1)[:, 1].cpu().detach().numpy().tolist()

        avg_loss, avg_acc = np.mean(losses), correct / total

        auc = roc_auc_score(true_labels, predicted_probs)
        balanced_acc = balanced_accuracy_score(true_labels, (np.array(predicted_probs) > 0.5).astype(int))

        writer.add_scalar("train_loss", avg_loss, epoch)
        writer.add_scalar("train_acc", avg_acc, epoch)
        writer.add_scalar("train_auc", auc, epoch)
        writer.add_scalar("train_balanced_acc", balanced_acc, epoch)

        print(
            f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f} train AUC {auc:.3f} train balanced acc {balanced_acc:.3f}")

        max_grad_norm = max(p.grad.data.norm(2).item() for p in model.parameters() if p.grad is not None)
        num_zeros = sum((p.grad.data == 0).sum().item() for p in model.parameters() if p.grad is not None)
        print(f"Max grad: {max_grad_norm}   Num zeros: {num_zeros}")

        # avg_loss, avg_acc = np.mean(losses), correct / total
        # writer.add_scalar("train_loss", avg_loss)
        # writer.add_scalar("train_acc", avg_acc)
        # print(f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f}")

        if epoch % 5 == 0:
            model.eval()
            losses, total, correct = [], 0, 0
            for imgs, lbls in val_gen.train_data():
                imgs = torch.Tensor(imgs).cuda()
                lbls = torch.Tensor(lbls).long().cuda()
                preds = model(imgs)

                loss = criterion(preds.view(-1, 2), lbls.view(-1))

                losses.append(loss.item())
                total += lbls.view(-1).shape[0]
                correct += (preds.view(-1, 2).argmax(dim=1) == lbls.view(-1)).sum().item()

                true_labels += lbls.view(-1).cpu().numpy().tolist()
                predicted_probs += torch.softmax(preds.view(-1, 2), dim=1)[:, 1].cpu().detach().numpy().tolist()

            avg_loss, avg_acc = np.mean(losses), correct / total

            auc = roc_auc_score(true_labels, predicted_probs)
            balanced_acc = balanced_accuracy_score(true_labels, (np.array(predicted_probs) > 0.5).astype(int))
            new_mean = (balanced_acc + auc) / 2
            if new_mean >= old_mean:
                torch.save(model.state_dict(), f"models/{model_name}/{model_name}_{epoch}.pth")
                old_mean = new_mean

            writer.add_scalar("test_loss", avg_loss, epoch)
            writer.add_scalar("test_acc", avg_acc, epoch)
            writer.add_scalar("test_auc", auc, epoch)
            writer.add_scalar("test_balanced_acc", balanced_acc, epoch)

            print(
                f"Epoch {epoch}: test loss {avg_loss:.3f} test acc {avg_acc:.3f} test AUC {auc:.3f} test balanced acc {balanced_acc:.3f}")
