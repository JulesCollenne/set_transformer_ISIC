import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, roc_curve

from data_isic import DataLoaderISIC
from models import SetTransformer

parser = argparse.ArgumentParser()
# parser.add_argument("--num_pts", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dim", type=int, default=256)
parser.add_argument("--n_heads", type=int, default=8)
parser.add_argument("--n_anc", type=int, default=16)
parser.add_argument("--train_epochs", type=int, default=150)
args = parser.parse_args()


def main():
    n_run = 4
    models = ("BYOL", "Moco", "SimCLR", "SimSiam", "SwaV")
    # models = ["reduced_CNN"]
    # models = ("Moco", "SimCLR", "SimSiam")
    # models = ("BYOL", "Moco")

    for model_name in models:
        print("Training:", model_name)
        args.exp_name = f"{model_name}Nd{args.dim}h{args.n_heads}i{args.n_anc}_lr{args.learning_rate}" \
                        f"bs{args.batch_size}"

        num_inds = 20
        n_classes = 2
        patience = 15

        do_train = True

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
        all_tpr, all_fpr = [], []
        base_fpr = np.linspace(0, 1, 101)

        for run in range(n_run):
            model = SetTransformer(n_features, num_inds, n_classes, num_inds=num_inds, num_heads=args.n_heads)
            model = nn.DataParallel(model)
            model = model.cuda()
            # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
            class_weights = torch.tensor([0.02, 0.98]).cuda()
            criterion = nn.CrossEntropyLoss(class_weights)
            old_mean = 0
            current_patience = 0

            if do_train:
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
                            predicted_probs += torch.softmax(preds.view(-1, 2), dim=1)[:,
                                               1].cpu().detach().numpy().tolist()

                        avg_loss, avg_acc = np.mean(losses), correct / total

                        auc = roc_auc_score(true_labels, predicted_probs)
                        balanced_acc = balanced_accuracy_score(true_labels,
                                                               (np.array(predicted_probs) > 0.5).astype(int))
                        # new_mean = (balanced_acc + auc) / 2
                        new_mean = auc

                        if new_mean >= old_mean:
                            torch.save(model.state_dict(), f"models/{model_name}/{model_name}_{run}_{num_inds}.pth")
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

            model = SetTransformer(n_features, num_inds, n_classes, num_inds=num_inds, num_heads=args.n_heads)
            model = nn.DataParallel(model)
            model = model.cuda()

            # checkpoint = torch.load(f"models/{model_name}/{model_name}.pth")
            # model.load_state_dict(checkpoint["state_dict"])
            model.load_state_dict(torch.load(f"models/{model_name}/{model_name}_{run}_{num_inds}.pth"))
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
            thresholds = np.linspace(0, 1, 5)

            for thresh in thresholds:
                balanced_acc = balanced_accuracy_score(true_labels, (np.array(predicted_probs) > thresh).astype(int))
                if balanced_acc > best_bacc:
                    best_bacc = balanced_acc
                    best_thresh = thresh

            print(
                f"val loss {avg_loss:.3f} val acc {avg_acc:.3f} val AUC {auc:.3f}"
                f" val balanced acc {best_bacc:.3f} best threshold {best_thresh:.3f}")

            losses, total, correct, true_labels, predicted_probs = [], 0, 0, [], []
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

            np.savetxt(f"predictions/{model_name}_{run}_{num_inds}", np.array(predicted_probs), delimiter=",")
            binary_predictions = (np.array(predicted_probs) > best_thresh).astype(int)

            conf_matrix = confusion_matrix(true_labels, binary_predictions)

            tn, fp, fn, tp = conf_matrix.ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            all_auc.append(roc_auc_score(true_labels, predicted_probs))
            all_bacc.append(balanced_accuracy_score(true_labels, binary_predictions))
            all_sens.append(sensitivity)
            all_spec.append(specificity)
            fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
            tpr = np.interp(base_fpr, fpr, tpr)
            tpr[0] = 0.0
            all_tpr.append(tpr)
            all_fpr.append(fpr)

            print(
                f"test loss {avg_loss:.3f} test acc {avg_acc:.3f} test AUC {all_auc[-1]:.3f}"
                f" test balanced acc {all_bacc[-1]:.3f}")

        print(model_name)
        print(
            f"test AUC {np.mean(all_auc):.3f} +/- {np.std(all_auc)} "
            f"test bacc {np.mean(all_bacc):.3f} +/- {np.std(all_bacc)} "
            f"test sensitivity {np.mean(all_sens):.3f} +/- {np.std(all_sens)} "
            f"test specificity {np.mean(all_spec):.3f} +/- {np.std(all_spec)}")

        plot_average_roc(all_tpr, base_fpr, all_auc, model_name)
    plt.savefig('roc_curve_with_std.png', dpi=300, bbox_inches='tight')


def plot_average_roc(all_tpr, base_fpr, all_auc, model_name):
    color = {"BYOL": 'blue',
             "Moco": 'green',
             "SwaV": 'red',
             "SimSiam": 'magenta',
             "SimCLR": 'indigo'}
    # mean_fpr = np.mean(all_fpr, axis=0)
    mean_tpr = np.mean(all_tpr, axis=0)
    std_tpr = np.std(all_tpr, axis=0)

    mean_auc = np.mean(all_auc)
    std_auc = np.std(all_auc)

    plt.figure(figsize=(8, 6))

    plt.plot(base_fpr, mean_tpr, color=color[model_name],
             label=f'{model_name} (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})')

    plt.fill_between(base_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='grey', alpha=0.3, label='Std Dev')

    # Plot ROC curve for each run (optional)
    # for i in range(len(all_fpr)):
    #     plt.plot(all_fpr[i], all_tpr[i], alpha=0.3)

    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Average ROC Curve with Standard Deviation')
    plt.legend(loc='lower right')
    plt.grid()


if __name__ == "__main__":
    main()
