import os

import numpy as np
import torch
import torchvision
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
import torchmetrics
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Set your dataset path
data_dir = "/home/jules.collenne/ISIC_2020/train/"
input_size = 300
batch_size = 64
num_workers = 8

# Define the data transforms for validation
val_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the test dataset
test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


# Define the ResNet-18 model
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.model = torchvision.models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


# Initialize the model
checkpoint_path = "/home/jules.collenne/ISIC_2020/lightly-ISIC/benchmarks/imagenet/resnet50/best18/resnet18_best.pth"

model = ResNetClassifier(num_classes=len(test_dataset.classes))
model.load_state_dict(torch.load(checkpoint_path))
model.to("cuda")  # Move the model to the GPU
model.eval()

# Lists to store true labels and predicted probabilities
true_labels = []
predicted_probs = []

# Iterate through the test loader
for batch in test_loader:
    x, y = batch
    x, y = x.to("cuda"), y.to("cuda")  # Move data to GPU
    logits = model(x)

    true_labels.extend(y.view(-1).cpu().numpy().tolist())
    predicted_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().detach().numpy().tolist())

# Compute AUC, balanced accuracy, sensitivity, and specificity
auc = roc_auc_score(true_labels, predicted_probs)
binary_predictions = (np.asarray(predicted_probs) >= 0.5).astype(int)
conf_matrix = confusion_matrix(true_labels, binary_predictions)
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
bacc = balanced_accuracy_score(true_labels, binary_predictions)

print(f"AUC: {auc:.3f}")
print(f"Balanced Accuracy: {bacc:.3f}")
print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
