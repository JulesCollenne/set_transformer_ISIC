import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from residualconvset import plot_average_roc, ResConvSet
from training import Training
from transformer_comparison import DataLoaderISIC


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


if __name__ == "__main__":
    main()
