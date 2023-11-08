from os.path import join
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.rotation import random_rotation_transform
from lightly.transforms.utils import IMAGENET_NORMALIZE
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from models import SetTransformer


def main():
    n_imgs_patient = 10
    img_size = 300
    n_classes = 2
    batch_size = 2
    epochs = 200
    lr = 1e-3
    feature_size = 2048

    dataset_path = "/home/jules.collenne/ISIC_2020/train/"
    outpath = "/home/jules.collenne/set_transformer_ISIC/ConvSetNet"

    train_gen, val_gen, tets_gen = get_datagen(dataset_path, batch_size, img_size, n_imgs_patient, feature_size)

    backbone = ResNetBackbone()

    set_transformer = SetTransformer(backbone.feature_size, n_imgs_patient, n_classes, num_inds=n_imgs_patient)

    # Create the ConvSetNet model with your specific number of classes and SetTransformer
    convsetnet = ConvSetNet(num_classes=n_classes, backbone=backbone, set_transformer=set_transformer)

    exp = Training(convsetnet, epochs, train_gen, val_gen, outpath, lr)
    exp.train()


# Define the modified ResNet-50 backbone
class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetBackbone, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.feature_size = 2048
        self.name = "ResNet-50"

    def forward(self, x):
        # Extract features from the ResNet-50 backbone
        features = self.resnet(x)
        return features


# Define the ConvSetNet model
class ConvSetNet(nn.Module):
    def __init__(self, num_classes, backbone, set_transformer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = backbone
        self.set_transformer = set_transformer
        self.final_classifier = nn.Linear(self.backbone.feature_size, num_classes)
        self.name = "ConvSetNet"

    def forward(self, images):
        # Get features from the backbone
        features = torch.stack([self.backbone(image) for image in images]).squeeze()

        # Pass the features to the SetTransformer (assuming set_transformer accepts features as input)
        set_transformer_output = self.set_transformer(features)

        # Make predictions through the final classifier
        # predictions = self.final_classifier(set_transformer_output)

        return set_transformer_output


class Training:
    def __init__(self, model, epochs, train_gen, val_gen, outpath, learning_rate):
        self.model = model
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.learning_rate = learning_rate
        self.epochs = epochs
        exp_name = f"{self.model.name}"
        log_dir = outpath + "result/" + exp_name
        self.writer = SummaryWriter(log_dir)

        self.class_weights = torch.tensor([0.02, 0.98]).cuda()
        self.old_mean = 0
        self.criterion = nn.CrossEntropyLoss(self.class_weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        print("Training", self.model.name)
        self.model = nn.DataParallel(self.model)
        self.model = self.model.cuda()
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            self.validate(epoch)

    def train_one_epoch(self, epoch):
        self.model.train()
        losses, total, correct, true_labels, predicted_probs = [], 0, 0, [], []

        for imgs, lbls in self.train_gen:
            imgs = torch.Tensor(imgs).cuda()
            lbls = torch.Tensor(lbls).long().cuda()
            preds = self.model(imgs)

            # loss = criterion(preds, lbls)
            loss = self.criterion(preds.view(-1, 2), lbls.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            total += lbls.view(-1).shape[0]
            correct += (preds.view(-1, 2).argmax(dim=1) == lbls.view(-1)).sum().item()

            true_labels += lbls.view(-1).cpu().numpy().tolist()
            predicted_probs += torch.softmax(preds.view(-1, 2), dim=1)[:, 1].cpu().detach().numpy().tolist()

        avg_loss, avg_acc = np.mean(losses), correct / total

        auc = roc_auc_score(true_labels, predicted_probs)
        balanced_acc = balanced_accuracy_score(true_labels, (np.array(predicted_probs) > 0.5).astype(int))

        self.writer.add_scalar("train_loss", avg_loss, epoch)
        self.writer.add_scalar("train_acc", avg_acc, epoch)
        self.writer.add_scalar("train_auc", auc, epoch)
        self.writer.add_scalar("train_balanced_acc", balanced_acc, epoch)

        print(
            f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f} train AUC {auc:.3f} train balanced acc {balanced_acc:.3f}")

        # ---------- In case of vanishing / exploding gradient ---------
        # max_grad_norm = max(p.grad.data.norm(2).item() for p in self.model.parameters() if p.grad is not None)
        # num_zeros = sum((p.grad.data == 0).sum().item() for p in self.model.parameters() if p.grad is not None)
        # print(f"Max grad: {max_grad_norm}   Num zeros: {num_zeros}")

        # avg_loss, avg_acc = np.mean(losses), correct / total
        # writer.add_scalar("train_loss", avg_loss)
        # writer.add_scalar("train_acc", avg_acc)
        # print(f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f}")

    def validate(self, epoch):
        self.model.eval()
        losses, total, correct, true_labels, predicted_probs = [], 0, 0, [], []
        for imgs, lbls in self.val_gen:
            imgs = torch.Tensor(imgs).cuda()
            lbls = torch.Tensor(lbls).long().cuda()
            preds = self.model(imgs)

            loss = self.criterion(preds.view(-1, 2), lbls.view(-1))

            losses.append(loss.item())
            total += lbls.view(-1).shape[0]
            correct += (preds.view(-1, 2).argmax(dim=1) == lbls.view(-1)).sum().item()

            true_labels += lbls.view(-1).cpu().numpy().tolist()
            predicted_probs += torch.softmax(preds.view(-1, 2), dim=1)[:, 1].cpu().detach().numpy().tolist()

        avg_loss, avg_acc = np.mean(losses), correct / total

        auc = roc_auc_score(true_labels, predicted_probs)
        balanced_acc = balanced_accuracy_score(true_labels, (np.array(predicted_probs) > 0.5).astype(int))
        new_mean = (balanced_acc + auc) / 2
        if new_mean >= self.old_mean:
            torch.save(self.model.state_dict(),
                       f"/home/jules.collenne/set_transformer_ISIC/ConvSetNet/models/{self.model.module.name}/{self.model.module.name}_{epoch}.pth")
            self.old_mean = new_mean

        self.writer.add_scalar("test_loss", avg_loss, epoch)
        self.writer.add_scalar("test_acc", avg_acc, epoch)
        self.writer.add_scalar("test_auc", auc, epoch)
        self.writer.add_scalar("test_balanced_acc", balanced_acc, epoch)

        print(
            f"Epoch {epoch}: test loss {avg_loss:.3f} test acc {avg_acc:.3f} test AUC {auc:.3f} test balanced acc "
            f"{balanced_acc:.3f}")


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, ipp, feature_size, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.ipp = ipp
        self.transform = transform
        self.patients = self.df['patient_id'].unique()
        self.feature_size = feature_size

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_ids = self.patients[idx]
        filtered_df = self.df[self.df['patient_id'] == patient_ids]
        targets = []
        images = []
        patient_target = filtered_df[filtered_df["patient_id"] == patient_ids]["target"].values
        patient_images = filtered_df[filtered_df["patient_id"] == patient_ids]["image_name"].values
        for i in range(self.ipp):
            if i < len(patient_images):
                images.append(
                    self.transform(Image.open(join(self.root_dir, "mel" if patient_target[i] == 1 else "nev",
                                                   patient_images[i] + ".jpg"))))
                targets.append(patient_target[i])
            else:
                images.append(torch.zeros(3, 300, 300))
                targets.append(0)

        data_tensor = torch.stack(images)
        targets_tensor = torch.tensor(targets)

        return data_tensor, targets_tensor


def get_datagen(dataset_path, batch_size, image_size, ipp, feature_size):
    transform = TrainTransform(input_size=image_size)

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset("/home/jules.collenne/ISIC_2020/y_train.csv", join(dataset_path, "train"), ipp=ipp,
                                  feature_size=feature_size, transform=transform)
    train_gen = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = CustomDataset("/home/jules.collenne/ISIC_2020/y_val.csv", join(dataset_path, "val"), ipp=ipp,
                                feature_size=feature_size, transform=transform)
    val_gen = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    test_dataset = CustomDataset("/home/jules.collenne/ISIC_2020/y_test.csv", join(dataset_path, "test"), ipp=ipp,
                                 feature_size=feature_size, transform=test_transform)
    test_gen = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_gen, val_gen, test_gen


class TrainTransform:
    def __init__(
            self,
            input_size: int = 300,
            cj_prob: float = 0.8,
            cj_strength: float = 1.0,
            cj_bright: float = 0.4,
            cj_contrast: float = 0.4,
            cj_sat: float = 0.4,
            cj_hue: float = 0.1,
            min_scale: float = 0.2,
            random_gray_scale: float = 0.2,
            gaussian_blur: float = 0.5,
            kernel_size: Optional[float] = None,
            sigmas: Tuple[float, float] = (0.1, 2),
            vf_prob: float = 0.0,
            hf_prob: float = 0.5,
            rr_prob: float = 0.0,
            rr_degrees: Union[None, float, Tuple[float, float]] = None,
            normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        color_jitter = T.ColorJitter(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
        )

        transform = [
            T.RandomResizedCrop(size=(input_size, input_size), scale=(min_scale, 1.0)),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
            T.ToTensor(),
        ]
        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image.Image]) -> Tensor:
        transformed: Tensor = self.transform(image)
        return transformed


if __name__ == "__main__":
    main()
