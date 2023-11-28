import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

from data_isic import DataLoaderISIC
from modules import ISAB, PMA, SAB

parser = argparse.ArgumentParser()
# parser.add_argument("--num_pts", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dim", type=int, default=256)
parser.add_argument("--n_heads", type=int, default=8)
parser.add_argument("--n_anc", type=int, default=16)
parser.add_argument("--train_epochs", type=int, default=150)
args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_layers=2, dropout=0., output_dim=2):
        super(Transformer, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(1, 0, 2)

        encoded = self.transformer_encoder(embedded)

        encoded = encoded.permute(1, 0, 2)  # Reshape to original dimensions
        output = self.output_layer(encoded)

        return output


def main():
    n_run = 4
    # models = ("BYOL", "Moco", "SimCLR", "SimSiam", "SwaV", "reduced_CNN")
    models = ["reduced_CNN"]
    # models = ("Moco", "SimCLR", "SimSiam")
    # models = ("BYOL", "Moco")

    do_training = True
    do_testing = True

    for model_name in models:
        print("Training:", model_name)
        args.exp_name = f"{model_name}Nd{args.dim}h{args.n_heads}i{args.n_anc}_lr{args.learning_rate}" \
                        f"bs{args.batch_size}"

        num_inds = 20
        n_classes = 2
        dim_inner_output = 2
        patience = 15

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

        n_features = sum(['feature' in col for col in pd.read_csv(f"features/{model_name}_val.csv").columns])

        all_auc, all_bacc, all_sens, all_spec = [], [], [], []

        class_weights = torch.tensor([0.02, 0.98]).cuda()
        criterion = nn.CrossEntropyLoss(class_weights)

        for run in range(n_run):
            outname = f"{model_name}/classical_{model_name}_{run}_{num_inds}_{dim_inner_output}.pth"
            # outname = f"{model_name}/residual_{model_name}_{run}_{num_inds}.pth"
            if do_training:
                model = Transformer(n_features, num_heads=args.n_heads,
                                    output_dim=n_classes)
                train(model, train_gen, val_gen, criterion, outname, patience)

            if do_testing:
                model = Transformer(n_features, num_heads=args.n_heads,
                                    output_dim=n_classes)

                model = nn.DataParallel(model)
                model = model.cuda()

                # checkpoint = torch.load(f"models/{model_name}/{model_name}.pth")
                # model.load_state_dict(checkpoint["state_dict"])
                model.load_state_dict(torch.load(f"models/{outname}"))
                model.eval()

                losses, total, correct, true_labels, predicted_probs = [], 0, 0, [], []
                for imgs, lbls in val_gen.train_data():
                    imgs = torch.Tensor(imgs).cuda()
                    lbls = torch.Tensor(lbls).long().cuda()
                    preds = model(imgs)

                    zero_rows_mask = torch.all(imgs == 0, dim=2)
                    non_zero_rows_mask = ~zero_rows_mask
                    preds = preds[non_zero_rows_mask]
                    lbls = lbls[non_zero_rows_mask]

                    loss = criterion(preds.view(-1, 2), lbls.view(-1))

                    losses.append(loss.item())
                    total += lbls.view(-1).shape[0]
                    correct += (preds.view(-1, 2).argmax(dim=1) == lbls.view(-1)).sum().item()

                    true_labels += lbls.view(-1).cpu().numpy().tolist()
                    predicted_probs += torch.softmax(preds.view(-1, 2), dim=1)[:, 1].cpu().detach().numpy().tolist()

                avg_loss, avg_acc = np.mean(losses), correct / total

                auc = roc_auc_score(true_labels, predicted_probs)
                best_bacc = 0
                best_thresh = 0
                thresholds = np.linspace(0, 1, 50)

                for thresh in thresholds:
                    balanced_acc = balanced_accuracy_score(true_labels, (np.array(predicted_probs) > thresh).astype(int))
                    if balanced_acc > best_bacc:
                        best_bacc = balanced_acc
                        best_thresh = thresh

                print(
                    f"val loss {avg_loss:.3f} val acc {avg_acc:.3f} val AUC {auc:.3f}"
                    f" val balanced acc {best_bacc:.3f} best threshold {best_thresh:.3f}")

                losses, total, correct, true_labels, predicted_probs = [], 0, 0, [], []
                # for imgs, lbls in test_gen.train_data():
                for imgs, lbls in test_gen.train_data():
                    imgs = torch.Tensor(imgs).cuda()
                    lbls = torch.Tensor(lbls).long().cuda()
                    preds = model(imgs)

                    zero_rows_mask = torch.all(imgs == 0, dim=2)
                    non_zero_rows_mask = ~zero_rows_mask
                    preds = preds[non_zero_rows_mask]
                    lbls = lbls[non_zero_rows_mask]

                    loss = criterion(preds.view(-1, 2), lbls.view(-1))

                    losses.append(loss.item())
                    total += lbls.view(-1).shape[0]
                    correct += (preds.view(-1, 2).argmax(dim=1) == lbls.view(-1)).sum().item()

                    true_labels += lbls.view(-1).cpu().numpy().tolist()
                    predicted_probs += torch.softmax(preds.view(-1, 2), dim=1)[:, 1].cpu().detach().numpy().tolist()

                avg_loss, avg_acc = np.mean(losses), correct / total

                np.savetxt(f"predictions/{outname}", np.array(predicted_probs), delimiter=",")
                binary_predictions = (np.array(predicted_probs) > best_thresh).astype(int)

                conf_matrix = confusion_matrix(true_labels, binary_predictions)

                tn, fp, fn, tp = conf_matrix.ravel()
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)

                all_auc.append(roc_auc_score(true_labels, predicted_probs))
                all_bacc.append(balanced_accuracy_score(true_labels, binary_predictions))
                all_sens.append(sensitivity)
                all_spec.append(specificity)

                print(
                    f"test loss {avg_loss:.3f} test acc {avg_acc:.3f} test AUC {all_auc[-1]:.3f}"
                    f" test balanced acc {all_bacc[-1]:.3f}")

        print(model_name)
        print(
            f"test AUC {np.mean(all_auc):.3f} +/- {np.std(all_auc)} "
            f"test bacc {np.mean(all_bacc):.3f} +/- {np.std(all_bacc)} "
            f"test sensitivity {np.mean(all_sens):.3f} +/- {np.std(all_sens)} "
            f"test specificity {np.mean(all_spec):.3f} +/- {np.std(all_spec)}")


def train(model, train_gen, val_gen, criterion, outname, patience):
    model = nn.DataParallel(model)
    model = model.cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    old_mean = 0
    current_patience = 0

    for epoch in range(args.train_epochs):
        model.train()
        losses, total, correct, true_labels, predicted_probs = [], 0, 0, [], []

        for imgs, lbls in train_gen.train_data():
            imgs = torch.Tensor(imgs).cuda()
            lbls = torch.Tensor(lbls).long().cuda()
            preds = model(imgs)

            # loss = criterion(preds, lbls)

            zero_rows_mask = torch.all(imgs == 0, dim=2)
            non_zero_rows_mask = ~zero_rows_mask
            preds = preds[non_zero_rows_mask]
            lbls = lbls[non_zero_rows_mask]

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

        if epoch % 1 == 0:
            print(
                f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f} train AUC {auc:.3f}"
                f" train balanced acc {balanced_acc:.3f}")

        # max_grad_norm = max(p.grad.data.norm(2).item() for p in model.parameters() if p.grad is not None)
        # num_zeros = sum((p.grad.data == 0).sum().item() for p in model.parameters() if p.grad is not None)
        # print(f"Max grad: {max_grad_norm}   Num zeros: {num_zeros}")

        if epoch % 1 == 0:
            model.eval()
            losses, total, correct, true_labels, predicted_probs = [], 0, 0, [], []
            for imgs, lbls in val_gen.train_data():
                imgs = torch.Tensor(imgs).cuda()
                lbls = torch.Tensor(lbls).long().cuda()
                preds = model(imgs)

                zero_rows_mask = torch.all(imgs == 0, dim=2)
                non_zero_rows_mask = ~zero_rows_mask
                preds = preds[non_zero_rows_mask]
                lbls = lbls[non_zero_rows_mask]

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
                torch.save(model.state_dict(), f"models/{outname}")
                old_mean = new_mean
                current_patience = 0
            else:
                current_patience += 1

            if current_patience >= patience:
                print("Stopping training.")
                break

            if epoch % 1 == 0:
                print(
                    f"Epoch {epoch}: val loss {avg_loss:.3f} val acc {avg_acc:.3f} val AUC {auc:.3f} "
                    f"val balanced acc {balanced_acc:.3f}")


if __name__ == "__main__":
    main()
