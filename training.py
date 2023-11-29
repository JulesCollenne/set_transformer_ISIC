import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, roc_curve

from residualconvset import plot_average_roc, ResConvSet
from transformer_comparison import DataLoaderISIC


les_best_thresholds = {
            "BYOL": [0.490, 0.490, 0.469, 0.469],
            "Moco": [0.510, 0.531, 0.510, 0.531],
            "SimCLR": [0.429, 0.490, 0.408, 0.388],
            "SimSiam": [0.510, 0.531, 0.510, 0.490],
            "SwaV": [0.551, 0.469, 0.490, 0.571],
        }


def main():
    num_inds = 20
    n_classes = 2
    dim_inner_output = 10
    n_heads = 8
    batch_size = 64

    # model definition
    model_declaration = lambda n_features: ResConvSet(n_features, num_inds, n_classes, num_inds=num_inds,
                                                      num_heads=n_heads,
                                                      dim_inner_output=dim_inner_output)

    # training
    exp = Experiment(model_declaration, batch_size)
    exp.run()


class Experiment:
    def __init__(self, model_declaration, batch_size):
        self.train = None
        self.model_declaration = model_declaration
        self.batch_size = batch_size
        self.train_gen = None
        self.val_gen = None
        self.test_gen = None

    def run(self):
        n_run = 4
        base_fpr = np.linspace(0, 1, 101)
        # models = ("BYOL", "Moco", "SimCLR", "SimSiam", "SwaV")
        models = ["reduced_CNN"]

        do_training = False
        do_testing = False
        visualize_preds = True
        assert visualize_preds and not (do_training or do_testing)

        plt.figure(figsize=(8, 6))

        for model_name in models:
            print("Training:", model_name)
            num_inds = 20
            n_classes = 2
            dim_inner_output = 10
            patience = 15

            n_features = sum(['feature' in col for col in pd.read_csv(f"features/{model_name}_val.csv").columns])

            self.train_gen, self.val_gen, self.test_gen = self.get_gens(model_name)

            class_weights = torch.tensor([0.02, 0.98]).cuda()
            self.criterion = nn.CrossEntropyLoss(class_weights)

            for run in range(n_run):
                outname = f"{model_name}/residual_{model_name}_{run}_{num_inds}_{dim_inner_output}.pth"
                model_declaration = self.model_declaration()

                train = Training(model_name, model_declaration, self.train_gen, self.val_gen, self.test_gen)

                all_auc, all_bacc, all_sens, all_spec, all_tpr, all_fpr = train.run(outname, do_training, do_testing,
                                                                                    visualize_preds)

                if visualize_preds:
                    print(model_name)
                    print(
                        f"test AUC {np.mean(all_auc):.3f} +/- {np.std(all_auc)} "
                        f"test bacc {np.mean(all_bacc):.3f} +/- {np.std(all_bacc)} "
                        f"test sensitivity {np.mean(all_sens):.3f} +/- {np.std(all_sens)} "
                        f"test specificity {np.mean(all_spec):.3f} +/- {np.std(all_spec)}")

                    plot_average_roc(all_tpr, base_fpr, all_auc, model_name)

            if visualize_preds:
                plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random Guessing')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Average ROC Curve with Standard Deviation')
                plt.legend(loc='lower right')
                plt.grid()
                plt.savefig('roc_curve_with_std.png', dpi=300, bbox_inches='tight')

    def get_gens(self, model_name):
        train_gen = DataLoaderISIC(
            f"features/{model_name}_train.csv",
            "GroundTruth.csv",
            batch_size=self.batch_size,
            input_dim=self.num_inds
        )

        val_gen = DataLoaderISIC(
            f"features/{model_name}_val.csv",
            "GroundTruth.csv",
            batch_size=self.batch_size,
            input_dim=self.num_inds
        )

        test_gen = DataLoaderISIC(
            f"features/{model_name}_test.csv",
            "GroundTruth.csv",
            batch_size=self.batch_size,
            input_dim=self.num_inds
        )
        return train_gen, val_gen, test_gen


class Training:
    def __init__(self, model_name, model_declaration, train_gen, val_gen, test_gen):
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.test_gen = test_gen
        self.model_declaration = model_declaration
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.learning_rate = 0.001
        self.train_epochs = 150
        self.batch_size = 64
        self.patience = 15
        self.base_fpr = np.linspace(0, 1, 101)
        self.model_name = model_name

    def run(self, outname, do_training, do_testing, visualize_preds):
        # outname = f"{model_name}/residual_{model_name}_{run}_{num_inds}.pth"
        all_auc, all_bacc, all_sens, all_spec = [], [], [], []
        all_tpr, all_fpr = [], []

        if do_training:
            model = self.model_declaration()
            self.train(model, outname, self.patience)

        if do_testing:
            model = self.model_declaration()
            model = nn.DataParallel(model)
            model = model.cuda()
            # checkpoint = torch.load(f"models/{model_name}/{model_name}.pth")
            # model.load_state_dict(checkpoint["state_dict"])
            model.load_state_dict(torch.load(f"models/{outname}"))
            model.eval()

            losses, total, correct, true_labels, predicted_probs = self.do_one_epoch(self.val_gen, False)

            avg_loss, avg_acc = np.mean(losses), correct / total
            auc = roc_auc_score(true_labels, predicted_probs)

            best_thresh, best_bacc = find_best_threshold(true_labels, predicted_probs)

            print(
                f"val loss {avg_loss:.3f} val acc {avg_acc:.3f} val AUC {auc:.3f}"
                f" val balanced acc {best_bacc:.3f} best threshold {best_thresh:.3f}")

            losses, total, correct, true_labels, predicted_probs = self.do_one_epoch(self.test_gen, False)

            avg_loss, avg_acc = np.mean(losses), correct / total

            data = pd.DataFrame({'p': [pred for pred in predicted_probs],
                                 'target': true_labels})
            # np.savetxt(f"predictions/{model_name}_{run}_{num_inds}", np.array(predicted_probs), delimiter=",")
            data.to_csv(f"predictions/{outname}.csv", index=False)
            binary_predictions = (np.array(predicted_probs) > best_thresh).astype(int)

            sensitivity, specificity, tpr, fpr = get_results(true_labels, binary_predictions, predicted_probs,
                                                             self.base_fpr)

            all_auc.append(roc_auc_score(true_labels, predicted_probs))
            all_bacc.append(balanced_accuracy_score(true_labels, binary_predictions))
            all_sens.append(sensitivity)
            all_spec.append(specificity)
            all_tpr.append(tpr)
            all_fpr.append(fpr)

            print(
                f"test loss {avg_loss:.3f} test acc {avg_acc:.3f} test AUC {all_auc[-1]:.3f}"
                f" test balanced acc {all_bacc[-1]:.3f}")

        if visualize_preds:
            data = pd.read_csv(f"predictions/{outname}.csv")
            true_labels = data["target"]
            predicted_probs = data["p"]

            best_thresh = les_best_thresholds[self.model_name][self.run]

            binary_predictions = (np.array(predicted_probs) > best_thresh).astype(int)

            sensitivity, specificity, tpr, fpr = get_results(true_labels, binary_predictions, predicted_probs,
                                                             self.base_fpr)

            all_auc.append(roc_auc_score(true_labels, predicted_probs))
            all_bacc.append(balanced_accuracy_score(true_labels, binary_predictions))
            all_sens.append(sensitivity)
            all_spec.append(specificity)
            all_tpr.append(tpr)
            all_fpr.append(fpr)

        return all_auc, all_bacc, all_sens, all_spec, all_tpr, all_fpr

    def train(self, outname, patience, print_freq=1, val_freq=1):
        self.model = nn.DataParallel(self.model)
        self.model = self.model.cuda()
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        old_mean = 0
        current_patience = 0

        for epoch in range(self.train_epochs):
            self.model.train()
            losses, total, correct, true_labels, predicted_probs = self.do_one_epoch(self.train_gen, True)
            avg_loss, avg_acc = np.mean(losses), correct / total

            auc = roc_auc_score(true_labels, predicted_probs)
            balanced_acc = balanced_accuracy_score(true_labels, (np.array(predicted_probs) > 0.5).astype(int))

            if epoch % print_freq == 0:
                print(
                    f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f} train AUC {auc:.3f}"
                    f" train balanced acc {balanced_acc:.3f}")

            # max_grad_norm = max(p.grad.data.norm(2).item() for p in model.parameters() if p.grad is not None)
            # num_zeros = sum((p.grad.data == 0).sum().item() for p in model.parameters() if p.grad is not None)
            # print(f"Max grad: {max_grad_norm}   Num zeros: {num_zeros}")

            if epoch % val_freq == 0:
                self.model.eval()
                losses, total, correct, true_labels, predicted_probs = self.do_one_epoch(self.val_gen, False)
                avg_loss, avg_acc = np.mean(losses), correct / total

                auc = roc_auc_score(true_labels, predicted_probs)
                balanced_acc = balanced_accuracy_score(true_labels, (np.array(predicted_probs) > 0.5).astype(int))
                new_mean = (balanced_acc + auc) / 2

                if new_mean >= old_mean:
                    torch.save(self.model.state_dict(), f"models/{outname}")
                    old_mean = new_mean
                    current_patience = 0
                else:
                    current_patience += 1

                if current_patience >= patience:
                    print("Stopping training.")
                    break

                print(
                    f"Epoch {epoch}: val loss {avg_loss:.3f} val acc {avg_acc:.3f} val AUC {auc:.3f} "
                    f"val balanced acc {balanced_acc:.3f}")

    def do_one_epoch(self, gen, is_training):
        if is_training:
            self.model.train()
        else:
            self.model.eval()

        losses, total, correct, true_labels, predicted_probs = [], 0, 0, [], []
        for imgs, lbls in gen.train_data():
            imgs = torch.Tensor(imgs).cuda()
            lbls = torch.Tensor(lbls).long().cuda()
            preds = self.model(imgs)

            zero_rows_mask = torch.all(imgs == 0, dim=2)
            non_zero_rows_mask = ~zero_rows_mask
            preds = preds[non_zero_rows_mask]
            lbls = lbls[non_zero_rows_mask]

            loss = self.criterion(preds.view(-1, 2), lbls.view(-1))

            if is_training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            losses.append(loss.item())
            total += lbls.view(-1).shape[0]
            correct += (preds.view(-1, 2).argmax(dim=1) == lbls.view(-1)).sum().item()

            true_labels += lbls.view(-1).cpu().numpy().tolist()
            predicted_probs += torch.softmax(preds.view(-1, 2), dim=1)[:, 1].cpu().detach().numpy().tolist()
        return losses, total, correct, true_labels, predicted_probs


def find_best_threshold(true_labels, predicted_probs):
    best_bacc = 0
    best_thresh = 0
    thresholds = np.linspace(0, 1, 50)

    for thresh in thresholds:
        balanced_acc = balanced_accuracy_score(true_labels,
                                               (np.array(predicted_probs) > thresh).astype(int))
        if balanced_acc > best_bacc:
            best_bacc = balanced_acc
            best_thresh = thresh
    return best_thresh, best_bacc


def get_results(true_labels, binary_predictions, predicted_probs, base_fpr):
    conf_matrix = confusion_matrix(true_labels, binary_predictions)
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
    tpr = np.interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    return sensitivity, specificity, tpr, fpr


if __name__ == "__main__":
    main()
