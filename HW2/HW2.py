# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 20:41:58 2024

@author: cdpss
"""

#%%ResNet18
import os
import random
import csv
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import torchvision.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
from argparse import Namespace
from tqdm import tqdm

config = Namespace(
    random_seed=42,
    BATCH=48,
    n_epoch=100,
    lr=1e-4,  
    weight_decay=1e-5,    
    ckpt_path='model.pth',
    patience=5,  
    factor=0.5,  
)

TRA_PATH = 'HW2/data/train/'
TST_PATH = 'HW2/data/test/'
LABEL_PATH = 'HW2/data/train.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(config.random_seed)
torch.cuda.manual_seed_all(config.random_seed)
random.seed(config.random_seed)
np.random.seed(config.random_seed)


class FaceExpressionDataset(Dataset):
    def __init__(self, img_path, label_path=None, tfm=T.ToTensor()):
        n_samples = len(os.listdir(img_path))
        if label_path is not None:
            self.images = [f'{img_path}/{i+7000}.jpg' for i in range(n_samples)]
            self.labels = pd.read_csv(label_path)['label'].values.tolist()
        else:
            self.images = [f'{img_path}/{i}.jpg' for i in range(n_samples)]
            self.labels = None
        self.tfm = tfm

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")  
        img = self.tfm(img)
        if self.labels is not None:
            lab = torch.tensor(self.labels[idx]).long()
            return img, lab
        else:
            return img

    def __len__(self):
        return len(self.images)


train_tfm = T.Compose([
    T.RandomHorizontalFlip(),  
    T.RandomRotation(15), 
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), 
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),  
    T.RandomResizedCrop(64, scale=(0.8, 1.0)),  
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])
eval_tfm = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


train_dataset = FaceExpressionDataset(TRA_PATH, LABEL_PATH, tfm=train_tfm)
train_len = int(0.8 * len(train_dataset))
valid_len = len(train_dataset) - train_len
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_len, valid_len])
test_dataset = FaceExpressionDataset(TST_PATH, tfm=eval_tfm)


train_loader = DataLoader(train_dataset, batch_size=config.BATCH, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH, shuffle=False)


class FaceExpressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 128)   
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return x

def train(model, train_loader, valid_loader, config):
    model.to(device)
    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.factor, patience=config.patience, verbose=True)

    best_acc = 0
    train_losses, valid_losses = [], []
    early_stop_counter = 0
    for epoch in range(config.n_epoch):
        model.train()
        train_loss, train_acc = 0, 0
        for img, lab in tqdm(train_loader):
            img, lab = img.to(device), lab.to(device)
            output = model(img)
            optimizer.zero_grad()
            loss = criteria(output, lab)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (torch.argmax(output, dim=-1) == lab).float().mean().item()
        train_loss, train_acc = train_loss / len(train_loader), train_acc / len(train_loader)
        train_losses.append(train_loss)
        print(f'Epoch: {epoch+1}/{config.n_epoch}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}')

        model.eval()
        valid_loss, valid_acc = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for img, lab in valid_loader:
                img, lab = img.to(device), lab.to(device)
                output = model(img)
                loss = criteria(output, lab)
                valid_loss += loss.item()
                valid_acc += (torch.argmax(output, dim=-1) == lab).float().mean().item()
                all_preds.extend(torch.argmax(output, dim=-1).cpu().tolist())
                all_labels.extend(lab.cpu().tolist())
        valid_loss, valid_acc = valid_loss / len(valid_loader), valid_acc / len(valid_loader)
        valid_losses.append(valid_loss)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f'Epoch: {epoch+1}/{config.n_epoch}, valid loss: {valid_loss:.4f}, valid acc: {valid_acc:.4f}, f1 score: {f1:.4f}')

        scheduler.step(valid_loss)

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), config.ckpt_path)
            print(f'== best valid acc: {best_acc:.4f} ==')
            early_stop_counter = 0  
        else:
            early_stop_counter += 1

        if early_stop_counter >= config.patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(config.ckpt_path))

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(valid_losses)), valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def draw_confusion_matrix(model, valid_loader):
    predictions, labels = [], []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for img, lab in tqdm(valid_loader):
            img = img.to(device)
            output = model(img)
            predictions += torch.argmax(output, dim=-1).tolist()
            labels += lab.tolist()

    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[str(i) for i in range(7)], yticklabels=[str(i) for i in range(7)])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

def test(model, test_loader):
    predictions = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for img in tqdm(test_loader):
            img = img.to(device)
            output = model(img)
            predictions += torch.argmax(output, dim=-1).tolist()
    with open('predict.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        for id, r in enumerate(predictions):
            writer.writerow([id, r])

model = FaceExpressionNet()
train(model, train_loader, valid_loader, config)
draw_confusion_matrix(model, valid_loader)

model.load_state_dict(torch.load('model.pth'))
test(model, test_loader)