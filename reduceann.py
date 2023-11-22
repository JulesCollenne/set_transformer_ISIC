import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, confusion_matrix
import pandas as pd
from tqdm import tqdm


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    do_train = False
    do_save = False
    do_test = True
    do_extraction = False

    # Define your data loaders
    train_dataset = MyDataset("features/CNN_train.csv")
    val_dataset = MyDataset("features/CNN_val.csv")
    test_dataset = MyDataset("features/CNN_test.csv")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Initialize the model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda" if torch.cuda.is_available() else "cpu")
    n_runs = 4
    all_auc, all_bacc, all_sens, all_spec = [], [], [], []

    for run in range(n_runs):
        outname = f"models/ClassificationCNN/ClassificationCNN_{run}.pth"

        model = MyNetwork().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        class_weights = torch.tensor([0.02, 0.98]).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # # Training loop
        if do_train:
            num_epochs = 100
            for epoch in range(num_epochs):
                # Training
                train_loss = train(model, train_loader, optimizer, criterion, device)

                # Validation
                val_loss, val_accuracy, val_auc, val_bacc, val_sens, val_spec = evaluate(model, val_loader, criterion,
                                                                                         device)

                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
                    f" auc: {val_auc:.4f} bacc: {val_bacc:.4f} sens: {val_sens:.4f} spec: {val_spec:.4f}")

        # Save the model
        if do_save:
            torch.save(model.state_dict(), outname)

        model.load_state_dict(torch.load(outname))

        # Test the model
        if do_test:
            predicted_probs, true_labels = evaluate(model, val_loader, criterion,device, give_preds=True)

            best_bacc = 0
            best_thresh = 0
            thresholds = np.linspace(0, 1, 5)

            for thresh in thresholds:
                balanced_acc = balanced_accuracy_score(true_labels, (np.array(predicted_probs) > thresh).astype(int))
                if balanced_acc > best_bacc:
                    best_bacc = balanced_acc
                    best_thresh = thresh

            test_loss, test_accuracy, test_auc, test_bacc, test_sens, test_spec = test(model, test_loader, criterion,
                                                                                       device, thresh=best_thresh)

            all_auc.append(test_auc)
            all_bacc.append(test_bacc)
            all_sens.append(test_sens)
            all_spec.append(test_spec)

            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
                  f" auc: {test_auc:.4f} bacc: {test_bacc:.4f} sens: {test_sens:.4f} spec: {test_spec:.4f}")

    print(
        f"test AUC {np.mean(all_auc):.3f} +/- {np.std(all_auc)} "
        f"test bacc {np.mean(all_bacc):.3f} +/- {np.std(all_bacc)} "
        f"test sensitivity {np.mean(all_sens):.3f} +/- {np.std(all_sens)} "
        f"test specificity {np.mean(all_spec):.3f} +/- {np.std(all_spec)}")

    ### Extract features
    if do_extraction:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        for loader, name in zip((train_loader, val_loader, test_loader), ("train", "val", "test")):
            features, targets, images_names = extract_features(model, loader, device)
            features = np.asarray(features)
            feature_columns = [f'feature_{i}' for i in range(features.shape[1])]
            df_results = pd.DataFrame({
                'target': targets,  # Convert tensor to list
                'image_name': images_names,
                **{col_name: feature_values for col_name, feature_values in zip(feature_columns, zip(*features))}
            })

            df_results.to_csv(f'features/reduced_CNN_{name}.csv', index=False)


# Define your neural network
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.dense_layer = nn.Linear(512, 200)
        self.prediction_layer = nn.Linear(200, 2)

    def forward(self, x):
        x = self.dense_layer(x)
        x = torch.relu(x)
        x = self.prediction_layer(x)
        return x


# Custom dataset class
class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.names = self.df["image_name"]
        self.feat_cols = [f"feature_{i}" for i in range(512)]
        self.features = self.df[self.feat_cols].values
        self.targets = self.df["target"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # features = torch.tensor(self.df[self.feat_cols].iloc[idx].values, dtype=torch.float32)
        # target = torch.tensor(self.df["target"].iloc[idx], dtype=torch.long)
        # img_name = self.df["image_name"].iloc[idx]
        features = self.features[idx]
        img_name = self.names[idx]
        target = self.targets[idx]
        return features, target, img_name


# Function to train the model
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for inputs, targets, _ in train_loader:
        inputs, targets = inputs.to(device).float(), targets.to(device).float()
        targets = targets.to(torch.long)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)


# Function to evaluate the model on the validation set
def evaluate(model, val_loader, criterion, device, give_preds=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            targets = targets.to(torch.long)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            probs = nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().detach().numpy()[:, 1])
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().detach().numpy())
            all_targets.extend(targets.cpu().detach().numpy())

    all_preds = np.asarray(all_preds)
    all_targets = np.asarray(all_targets)
    if give_preds:
        return all_preds, all_targets
    accuracy = accuracy_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)
    balanced_acc = balanced_accuracy_score(all_targets, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return running_loss / len(val_loader), accuracy, auc, balanced_acc, sensitivity, specificity


# Function to test the model on the test set
def test(model, test_loader, criterion, device, thresh=None):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            targets = targets.to(torch.long)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            probs = nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().detach().numpy()[:, 1])
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().detach().numpy())
            all_targets.extend(targets.cpu().detach().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs)
    if thresh is not None:
        all_preds = (np.array(all_preds) > thresh).astype(int)
    balanced_acc = balanced_accuracy_score(all_targets, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return running_loss / len(test_loader), accuracy, auc, balanced_acc, sensitivity, specificity


def extract_features(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_names = []

    with torch.no_grad():
        for inputs, targets, names in test_loader:
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            targets = targets.to(torch.long)
            outputs = model.dense_layer(inputs)
            preds = torch.relu(outputs)

            all_preds.extend(preds.cpu().detach().numpy())
            all_targets.extend(targets.cpu().detach().numpy())
            all_names.extend(names)

    return all_preds, all_targets, all_names


if __name__ == "__main__":
    main()
