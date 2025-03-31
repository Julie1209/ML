#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 21:34:21 2024

@author: md703
"""

import os
import random
import csv
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision.transforms as T
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from argparse import Namespace
from tqdm import tqdm

# Load unlabelled data
def load_unlabelled(img_dir):
    return [os.path.join(img_dir, img) for img in os.listdir(img_dir) if img.endswith('.jpg')]

# Load labelled data
def load_labelled(root_dir):
    data = []
    labels = []
    for label in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label)
        if os.path.isdir(label_path):
            for img in os.listdir(label_path):
                if img.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(label_path, img)
                    data.append(img_path)
                    labels.append(int(label))
    return list(zip(data, labels))

# Image Dataset class
class ImageDataset(Dataset):
    def __init__(self, img_paths, labels=None, tfm=T.Compose([T.Resize((64, 64)), T.ToTensor()])):
        super().__init__()
        self.img_paths = img_paths
        self.labels = labels
        self.tfm = tfm

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img = self.tfm(img)
        if self.labels is None:
            return img
        else:
            return img, self.labels[idx]

    def __len__(self):
        return len(self.img_paths)

# Residual Block class
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

# Modified Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.enc1 = ResidualBlock(3, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)
        self.enc4 = ResidualBlock(256, 512)

        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(512, 1024)

        # Decoder with skip connections
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(128, 64)

        # Final convolution
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # Use sigmoid to get pixel values between 0 and 1

        # Classifier head with Dropout for regularization
        self.predictor = nn.Sequential(
            nn.Linear(1024 * 4 * 4, 1024),  # Adjust according to your latent space size
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout to reduce overfitting
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        b_flat = b.view(b.size(0), -1)

        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        # Final output
        x_prime = self.final_conv(d1)
        x_prime = self.sigmoid(x_prime)

        # Classifier
        y = self.predictor(b_flat)

        return x_prime, y, b_flat

# Custom loss function
def loss_fn(x_prime, x, y_hat, y, recon_weight=0.5):
    # Reconstruction Loss (MSE Loss)
    reconstruction_loss = nn.MSELoss()(x_prime, x)
    # Classification Loss (Cross Entropy Loss)
    classification_loss = nn.CrossEntropyLoss()(y_hat, y)
    # Combined Loss
    return recon_weight * reconstruction_loss + (1 - recon_weight) * classification_loss

# Function to add noise to images
def add_noise(x, noise_factor=0.1):
    noise = noise_factor * torch.randn_like(x)
    return x + noise

# Self-supervised pretraining function
def pretrain(model, train_loader, valid_loader, config, noise_function=add_noise):
    model = model.to(config.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_valid_loss = float('inf')
    for epoch in range(config.pretrain_epochs):
        model.train()
        train_loss = 0
        for img in tqdm(train_loader):
            img = img.to(config.device)
            optimizer.zero_grad()
            output, _, _ = model(noise_function(img, config.noise_factor))
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        if valid_loader is not None:
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for img in valid_loader:
                    img = img.to(config.device)
                    output, _, _ = model(img)
                    loss = criterion(output, img)
                    valid_loss += loss.item()
                valid_loss /= len(valid_loader)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), config.pretrain_model_path)

            print(f'Epoch {epoch+1}/{config.pretrain_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

# Fine-tune function
def finetune(model, train_loader, valid_loader, config, noise_function=add_noise):
    model = model.to(config.device)
    criterion = loss_fn
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_valid_loss = float('inf')
    for epoch in range(config.finetune_epochs):
        model.train()
        train_loss, train_acc = 0, 0
        for img, label in tqdm(train_loader):
            img, label = img.to(config.device), label.to(config.device)
            optimizer.zero_grad()
            # Forward pass with noise added to the image
            x_prime, y_hat, _ = model(noise_function(img, config.noise_factor))
            loss = criterion(x_prime, img, y_hat, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (y_hat.argmax(dim=1) == label).float().mean().item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        print(f'Epoch {epoch+1}/{config.finetune_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

        model.eval()
        valid_loss, valid_acc = 0, 0
        with torch.no_grad():
            for img, label in valid_loader:
                img, label = img.to(config.device), label.to(config.device)
                # Forward pass without noise for validation
                x_prime, y_hat, _ = model(img)
                loss = criterion(x_prime, img, y_hat, label)
                valid_loss += loss.item()
                valid_acc += (y_hat.argmax(dim=1) == label).float().mean().item()

            valid_loss /= len(valid_loader)
            valid_acc /= len(valid_loader)

        scheduler.step(valid_loss)

        # Save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), config.finetune_model_path)

        print(f'Epoch {epoch+1}/{config.finetune_epochs}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')

# Data transformations for pretraining and finetuning
pretrain_train_tfm = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.RandomResizedCrop(size=(64, 64)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

pretrain_valid_tfm = T.Compose([
    T.Resize(size=(64, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

pretrain_config = Namespace(
    batch_size=128,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    lr=1e-3,
    weight_decay=1e-4,
    noise_factor=0.01,
    pretrain_epochs=30,
    pretrain_model_path='pretrain_model.ckpt'
)

# Pretraining data loaders
pretrain_train_data = load_unlabelled('data/dev/unlabelled')
train_size = int(0.8 * len(pretrain_train_data))
valid_size = len(pretrain_train_data) - train_size
pretrain_train_data, pretrain_valid_data = torch.utils.data.random_split(pretrain_train_data, [train_size, valid_size])
pretrain_train_dataset = ImageDataset(pretrain_train_data, tfm=pretrain_train_tfm)
pretrain_valid_dataset = ImageDataset(pretrain_valid_data, tfm=pretrain_valid_tfm)

pretrain_train_loader = DataLoader(pretrain_train_dataset, batch_size=pretrain_config.batch_size, shuffle=True)
pretrain_valid_loader = DataLoader(pretrain_valid_dataset, batch_size=pretrain_config.batch_size, shuffle=False)

# Initialize and pretrain model
model = Autoencoder()
pretrain(model, pretrain_train_loader, pretrain_valid_loader, pretrain_config)

# Data transformations for finetuning
finetune_train_tfm = T.Compose([
    T.RandomResizedCrop((64, 64)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.RandomApply([T.GaussianBlur(3)], p=0.5),
    T.RandomRotation(degrees=15),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

finetune_valid_tfm = T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

finetune_config = Namespace(
    batch_size=16,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    lr=1e-4,
    weight_decay=1e-5,
    noise_factor=0.01,
    finetune_epochs=60,
    finetune_model_path='finetune_model.ckpt'
)

# Finetuning data loaders
finetune_train_data = load_labelled('data/dev/labelled')
train_size = int(0.8 * len(finetune_train_data))
valid_size = len(finetune_train_data) - train_size
finetune_train_data, finetune_valid_data = torch.utils.data.random_split(finetune_train_data, [train_size, valid_size])
finetune_train_dataset = ImageDataset(*map(list, zip(*finetune_train_data)), finetune_train_tfm)
finetune_valid_dataset = ImageDataset(*map(list, zip(*finetune_valid_data)), finetune_valid_tfm)

finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=finetune_config.batch_size, shuffle=True)
finetune_valid_loader = DataLoader(finetune_valid_dataset, batch_size=finetune_config.batch_size, shuffle=False)

# Load pre-trained model and fine-tune
model.load_state_dict(torch.load(pretrain_config.pretrain_model_path))
finetune(model, finetune_train_loader, finetune_valid_loader, finetune_config)

# Test the model
test_dataset = ImageDataset(load_unlabelled('data/test'), tfm=pretrain_valid_tfm)
test_loader = DataLoader(test_dataset, batch_size=finetune_config.batch_size, shuffle=False)

model.load_state_dict(torch.load(finetune_config.finetune_model_path))
model = model.to(finetune_config.device).eval()

# Generate predictions
predictions = []
with torch.no_grad():
    for img in test_loader:
        img = img.to(finetune_config.device)
        _, y, _ = model(img)
        predictions.append(y.argmax(dim=1).cpu().numpy())
predictions = np.concatenate(predictions)

# Save predictions
with open('predict.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'label'])
    for idx, label in enumerate(predictions):
        writer.writerow([idx, label])

from sklearn.manifold import TSNE
def Eq38_compute_weights(X, centroids, alpha):
    distances = np.linalg.norm(X[:, None] - centroids, axis=2) 
    weights = np.exp(-alpha * distances)  
    weights /= np.sum(weights, axis=1, keepdims=True)  
    return weights

def Eq39_update_centroids(X, weights):
    centroids = (weights.T @ X) / np.sum(weights, axis=0)[:, np.newaxis]  
    return centroids


def equilibrium_k_means(X, k, alpha, n_iter):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for _ in range(n_iter):

        weights = Eq38_compute_weights(X, centroids, alpha)
 
        centroids = Eq39_update_centroids(X, weights)
    
    return centroids

def select_clusters(weights):
    return np.argmax(weights, axis=1)

def prune_dimension(X):
    tsne = TSNE(n_components=2, random_state=42)
    Y = tsne.fit_transform(X)
    return Y

def plot_clusters(Y, centroids, labels, title='Clustering'):
    unique_labels = np.unique(labels)
    plt.figure(figsize=(10, 6))

    for label in unique_labels:
        plt.scatter(Y[labels == label, 0], Y[labels == label, 1], label=f'Cluster {label}', alpha=0.6)

    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

X, label = [], []
with torch.no_grad():
    for img, lab in finetune_valid_dataset:
        _, probs, latent_embedding = model(img.unsqueeze(0).to(finetune_config.device))
        X.append(latent_embedding.cpu().numpy())
        label.append(lab)
X, label = np.concatenate(X), np.array(label)

cls1, cls2, cls3 = 2, 5, 8
cls_idx = np.where((label == cls1) | (label == cls2) | (label == cls3))
X = X[cls_idx]
label = label[cls_idx]

target_ratio = np.array([2, 1, 1]) / 4
best_alpha = None
best_counts = None
best_centroids = None
best_clustering = None
best_diff = float('inf')

for alpha in np.arange(0.1, 10.0, 0.1):  
    centroids = equilibrium_k_means(X, k=3, alpha=alpha, n_iter=50)
    weights = Eq38_compute_weights(X, centroids, alpha)
    clustering = select_clusters(weights)
    cluster_counts = np.bincount(clustering, minlength=3)

    current_ratio = cluster_counts / np.sum(cluster_counts)
    diff = np.linalg.norm(current_ratio - target_ratio)

    if diff < best_diff:
        best_diff = diff
        best_alpha = alpha
        best_counts = cluster_counts
        best_centroids = centroids
        best_clustering = clustering


print("\n找到的最佳 alpha 值：", best_alpha)
print("每個群組的數據點數量：", best_counts)
print("比例：", best_counts / np.sum(best_counts))


total_data = np.concatenate([X, best_centroids])
Y = prune_dimension(total_data)
Y, centroids = Y[:-3], Y[-3:]
plot_clusters(Y, centroids, best_clustering, title=f'Equilibrium K-means Clustering after t-SNE (alpha={best_alpha:.2f})')

for alpha_multiplier in [10, 0.1]:
    new_alpha = best_alpha * alpha_multiplier
    centroids = equilibrium_k_means(X, k=3, alpha=new_alpha, n_iter=50)
    weights = Eq38_compute_weights(X, centroids, new_alpha)
    clustering = select_clusters(weights)

    total_data = np.concatenate([X, centroids])
    Y = prune_dimension(total_data)
    Y, centroids = Y[:-3], Y[-3:]

    plot_clusters(Y, centroids, clustering, title=f'Equilibrium K-means Clustering after t-SNE (alpha={new_alpha:.2f})')
    
    class1_dataset = ImageDataset(load_unlabelled('./data/dev/labelled/2/'))
class1_loader = DataLoader(class1_dataset, batch_size=8, shuffle=True)
model = Autoencoder()
pretrain(model, class1_loader, None, Namespace(
    device = 'cuda' if torch.cuda.is_available() else 'cpu',
    lr = pretrain_config.lr,
    weight_decay = 0,
    noise_factor = 0,
    pretrain_epochs = 50
))


anomaly_dataset = ImageDataset(load_unlabelled('data/anomoly'), tfm=T.Compose([
    T.Grayscale(num_output_channels=3),  
    T.ToTensor()
]))
anomaly_loader = DataLoader(anomaly_dataset, batch_size=pretrain_config.batch_size, shuffle=False)

anomaly_loss = []
with torch.no_grad():
    for image in anomaly_loader:
        image = image.to(pretrain_config.device)
        recon, _, _ = model(image)
        loss = nn.MSELoss()(recon, image)
        anomaly_loss.append(loss.item())
anomaly_loss = sum(anomaly_loss) / len(anomaly_loss)
print('Anomaly loss:', anomaly_loss)

class1_loss = []
with torch.no_grad():
    for image in class1_loader:
        image = image.to(pretrain_config.device)
        recon, _, _ = model(image)
        loss = nn.MSELoss()(recon, image)
        class1_loss.append(loss.item())
class1_loss = sum(class1_loss) / len(class1_loss)
print('Normal loss :', class1_loss)

idx1, idx2 = 42, 10
image1, image2 = anomaly_dataset[idx1], class1_dataset[idx2]
model.eval()
with torch.no_grad():
    recon1 = model(image1.unsqueeze(0).to(pretrain_config.device))[0].cpu().numpy()
    recon2 = model(image2.unsqueeze(0).to(pretrain_config.device))[0].cpu().numpy()

fig, axs = plt.subplots(2, 2, figsize=(10,10))
axs[0, 0].imshow(image1.numpy().transpose((1, 2, 0)))
axs[0, 0].set_title('Anomaly Image')
axs[0, 1].imshow(recon1.squeeze().transpose((1, 2, 0)))
axs[0, 1].set_title('Reconstructed Anomaly Image')
axs[1, 0].imshow(image2.numpy().transpose((1, 2, 0)))
axs[1, 0].set_title('Normal Image')
axs[1, 1].imshow(recon2.squeeze().transpose((1, 2, 0)))
axs[1, 1].set_title('Reconstructed Normal Image')
plt.show()
