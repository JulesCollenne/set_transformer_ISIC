import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

from data_isic import DataLoaderISIC
from models import SetTransformer

parser = argparse.ArgumentParser()
# parser.add_argument("--num_pts", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dim", type=int, default=256)
parser.add_argument("--n_heads", type=int, default=4)
parser.add_argument("--n_anc", type=int, default=16)
parser.add_argument("--train_epochs", type=int, default=100)
args = parser.parse_args()


def main():
    n_run = 2
    # models = ("BYOL", "Moco", "SimCLR", "SimSiam", "SwaV")
    models = ["CNN"]
    # models = ("Moco", "SimCLR", "SimSiam")
    # models = ("BYOL", "Moco")

    for model_name in models:
        print("Training:", model_name)
        args.exp_name = f"{model_name}Nd{args.dim}h{args.n_heads}i{args.n_anc}_lr{args.learning_rate}" \
                        f"bs{args.batch_size}"

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

        n_features = sum(['feature' in col for col in pd.read_csv(f"features/{model_name}_val.csv").columns])

        all_auc, all_bacc, all_sens, all_spec = [], [], [], []

        model = SetTransformer(n_features, 10, 2, num_inds=num_inds)
        model = nn.DataParallel(model)
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        class_weights = torch.tensor([0.02, 0.98]).cuda()
        criterion = nn.CrossEntropyLoss(class_weights)
        old_mean = 0

        model = SetTransformer(n_features, 10, 2, num_inds=num_inds)
        model = nn.DataParallel(model)
        model = model.cuda()

        # checkpoint = torch.load(f"models/{model_name}/{model_name}.pth")
        # model.load_state_dict(checkpoint["state_dict"])
        model.load_state_dict(torch.load(f"models/{model_name}/{model_name}.pth"))
        model.eval()

        losses, total, correct = [], 0, 0
        losses, total, correct, true_labels, predicted_probs = [], 0, 0, [], []
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

        auc = roc_auc_score(true_labels, predicted_probs)
        best_bacc = 0
        best_thresh = 0
        thresholds = np.linspace(0, 1, 100)

        for thresh in thresholds:
            balanced_acc = balanced_accuracy_score(true_labels, (np.array(predicted_probs) > thresh).astype(int))
            if balanced_acc > best_bacc:
                best_bacc = balanced_acc
                best_thresh = thresh

        print(
            f"val AUC {auc:.3f}"
            f" val balanced acc {best_bacc:.3f} best threshold {best_thresh:.3f}")

        losses, total, correct, true_labels, predicted_probs = [], 0, 0, [], []
        for imgs, lbls in test_gen.train_data():
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

        np.savetxt(f"predictions/{model_name}0", np.array(predicted_probs), delimiter=",")
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
            f" test balanced acc {all_bacc[-1]:.3f} test sens {all_sens[-1]:.3f} test spe {all_spec[-1]:.3f}")


if __name__ == "__main__":
    main()
