import os

import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import torchmetrics
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

# Set your dataset path
data_dir = "/media/jules/Transcend/Datasets/isic/ISIC_2020/train3/"
input_size = 300
batch_size = 64
num_epochs = 200
threshold = 0.5  # Define your threshold value
num_workers = 8

# Define the data transforms for training and validation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=val_transform)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
val_dataset = ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)

# Create PyTorch data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


# Define the ResNet-18 model
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.model = torchvision.models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor([0.02, 0.98]))
        self.accuracy = torchmetrics.Accuracy(task="binary", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def extract_features(self, x):
        # Get the output of the penultimate layer
        features = self.model.conv1(x)
        features = self.model.bn1(features)
        features = self.model.relu(features)
        features = self.model.maxpool(features)

        features = self.model.layer1(features)
        features = self.model.layer2(features)
        features = self.model.layer3(features)
        features = self.model.layer4(features)

        features = self.model.avgpool(features)
        features = torch.flatten(features, 1)

        return features


# Initialize the model
checkpoint_path = "resnet18_best.pth"

model = ResNetClassifier(num_classes=len(test_dataset.classes))
model.load_state_dict(torch.load(checkpoint_path))

model.to("cuda")  # Move the model to the GPU

for loader, name in zip([val_loader, test_loader], ["val", "test"]):
    train_features_list, val_features_list, test_features_list = [], [], []
    train_labels_list, val_labels_list, test_labels_list = [], [], []
    image_names_list = [os.path.basename(path) for path, _ in loader.dataset.imgs]
    for i, batch in enumerate(loader):
        x, y = batch
        x, y = x.to("cuda"), y.to("cuda")  # Move data to GPU
        features = model.extract_features(x).cpu().detach().numpy()  # Extract features and convert to NumPy array

        test_features_list.append(features)
        test_labels_list.append(y.cpu().detach().numpy())

    print("jai fini, cool")
    test_features = np.concatenate(test_features_list, axis=0)
    test_labels = np.concatenate(test_labels_list, axis=0)

    test_df = pd.DataFrame(data=test_features, columns=[f'feature_{i}' for i in range(test_features.shape[1])])
    test_df['label'] = test_labels
    test_df['image_name'] = image_names_list
    test_df.to_csv(f'CNN_{name}.csv', index=False)

print("bon la jai créé des fichiers mec")


