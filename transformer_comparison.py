import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 5 * 50, 128)
        self.fc2 = nn.Linear(128, 20)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 50)  # Reshape before fully connected layer
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define a custom dataset
class DataLoaderISIC:
    def __init__(self, df_name, gt_name, batch_size, input_dim=20):
        self.batch_size = batch_size
        self.input_dim = input_dim

        self.gt = pd.read_csv(gt_name)
        self.train_features = pd.read_csv(df_name)

        matching_patient_ids = self.gt[self.gt["image_name"].isin(self.train_features["image_name"])]["patient_id"]
        self.patients = matching_patient_ids.unique()
        self.n_features = sum(['feature' in col for col in self.train_features.columns])

    def __len__(self):
        return len(self.patients) // self.batch_size

    def train_data(self):
        start = 0
        end = self.batch_size
        while end < len(self.patients):
            patients_names = self.patients[start:end]
            patients_imgs = [self.gt[self.gt["patient_id"] == pid]["image_name"].values for pid in patients_names]
            current_features = []
            current_labels = []
            for pnum, current_patient_imgs in enumerate(patients_imgs):
                random.shuffle(current_patient_imgs)
                current_patient_features = []
                current_patient_labels = []
                for i in range(self.input_dim):
                    if i < len(current_patient_imgs):  # If a patient has more images than needed
                        current_patient_features.append(
                            self.train_features[self.train_features["image_name"] == current_patient_imgs[i]].filter(like="feature").values[0])
                        current_patient_labels.append(self.gt[self.gt["image_name"] == current_patient_imgs[i]]["target"].values[0])
                        # current_patient_labels.append(np.eye(2)[self.gt[self.gt["image_name"] == images[i]]["target"].values[0]])
                    else:
                        current_patient_features.append(np.zeros(self.n_features))
                        current_patient_labels.append(0)  # TODO not sre whether I should do something else
                        # current_patient_labels.append([1, 0])
                current_features.append(current_patient_features)
                current_labels.append(current_patient_labels)
            current_features = np.asarray(current_features)
            current_labels = np.asarray(current_labels)
            yield current_features, current_labels
            end += self.batch_size
            start += self.batch_size


def main():
    batch_size = 32
    n_patients = 20
    model_name = "reduced_CNN"

    train_gen = DataLoaderISIC(
                f"features/{model_name}_train.csv",
                "GroundTruth.csv",
                batch_size=batch_size,
                input_dim=n_patients
            )

    val_gen = DataLoaderISIC(
        f"features/{model_name}_val.csv",
        "GroundTruth.csv",
        batch_size=batch_size,
        input_dim=n_patients
    )

    test_gen = DataLoaderISIC(
        f"features/{model_name}_test.csv",
        "GroundTruth.csv",
        batch_size=batch_size,
        input_dim=n_patients
    )

    # Initialize the model, loss function, and optimizer
    model = CNNModel().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100

    total_loss = 0
    y_true = []
    y_scores = []

    for epoch in range(num_epochs):
        for images, labels in train_gen.train_data():
            optimizer.zero_grad()

            imgs = torch.Tensor(images).cuda().unsqueeze(1)
            lbls = torch.Tensor(labels).long().cuda().float().view(-1)
            preds = model(imgs)
            preds = preds.view(-1)

            zero_rows_mask = torch.all(imgs == 0, dim=-1)
            non_zero_rows_mask = ~zero_rows_mask
            non_zero_rows_mask = non_zero_rows_mask.view(-1)
            preds = preds[non_zero_rows_mask]
            lbls = lbls[non_zero_rows_mask]

            loss = criterion(preds, lbls.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            y_true.extend(lbls.cpu().numpy())
            y_scores.extend(torch.sigmoid(preds).detach().cpu().numpy())

        auc = roc_auc_score(y_true, y_scores)

        average_loss = total_loss / len(train_gen)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}, AUC: {auc:.4f}')
        old_mean = 0

        if epoch % 1 == 0:
            model.eval()
            losses, total, correct, true_labels, predicted_probs = [], 0, 0, [], []
            for imgs, lbls in val_gen.train_data():
                imgs = torch.Tensor(imgs).cuda().unsqueeze(1)
                lbls = torch.Tensor(lbls).long().cuda().float().view(-1)
                preds = model(imgs)
                preds = preds.view(-1)

                zero_rows_mask = torch.all(imgs == 0, dim=-1)
                non_zero_rows_mask = ~zero_rows_mask
                non_zero_rows_mask = non_zero_rows_mask.view(-1)
                preds = preds[non_zero_rows_mask]
                lbls = lbls[non_zero_rows_mask]

                loss = criterion(preds, lbls.view(-1))

                losses.append(loss.item())
                total += lbls.view(-1).shape[0]
                correct += ((preds > 0.5) == lbls.view(-1)).sum().item() #

                true_labels += lbls.view(-1).cpu().numpy().tolist()
                # predicted_probs += torch.softmax(preds.view(-1, 2), dim=1)[:, 1].cpu().detach().numpy().tolist()
                predicted_probs += preds.cpu().detach().numpy().tolist()

            avg_loss, avg_acc = np.mean(losses), correct / total

            auc = roc_auc_score(true_labels, predicted_probs)
            balanced_acc = balanced_accuracy_score(true_labels, (np.array(predicted_probs) > 0.5).astype(int))
            new_mean = auc
            if new_mean >= old_mean:
                torch.save(model.state_dict(), f"models/{model_name}/SimpleCNN_{model_name}.pth")
                old_mean = new_mean

            if epoch % 1 == 0:
                print(
                    f"Epoch {epoch}: val loss {avg_loss:.3f} val acc {avg_acc:.3f} val AUC {auc:.3f} "
                    f"val balanced acc {balanced_acc:.3f}")


if __name__ == "__main__":
    main()
