# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 23:17:00 2024

@author: cdpss
"""

import numpy as np
import csv
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
import matplotlib.pyplot as plt

from tqdm import tqdm
from argparse import Namespace
from sklearn.preprocessing import StandardScaler
from torchmetrics import AUROC

config = Namespace(
    num_epochs=50,
    lr=1e-4,  
    weight_decay=1e-5,  # 加入 weight decay 以提高泛化能力
    batch_size=256,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ckpt_path='model.ckpt'
)

"""## Data Processing"""

data_dir = '/media/md703/PNY_Gen4_4TB/Julie/HW4/HW4/data'

def handle_missing_values(df):
    # 根據每個欄位的數據特性來處理遺漏值
    for column in df.columns:
        if pd.api.types.is_integer_dtype(df[column]):
            # 用眾數來填充整數型欄位
            df[column] = df[column].fillna(df[column].mode()[0])
        elif pd.api.types.is_float_dtype(df[column]):
            # 用中位數來填充浮點型欄位
            df[column] = df[column].fillna(df[column].median())
        elif pd.api.types.is_object_dtype(df[column]):
            # 填充字符串/物件型欄位為 'Unknown'
            df[column] = df[column].fillna('Unknown')

# 加載數據
def load_csv(file_name, labelled=True, categories=None):
    # csv 文件沒有標頭
    df = pd.read_csv(file_name, header=None)

    # 如果是有標籤的數據，則刪除標籤缺失的行
    if labelled:
        lab = df[0].values
        df.drop(columns=0, inplace=True)
    else:
        lab = None

    # 使用 categories 將字符串特徵轉換為整數
    if categories is not None:
        i = 0
        for col in df.columns:
            if df[col].dtype == 'object' or pd.api.types.is_integer_dtype(df[col]):
                df[col] = pd.Categorical(df[col], categories=categories[i])
                df[col] = df[col].cat.codes
                i += 1
    else:
        categories = []
        for col in df.columns:
            if df[col].dtype == 'object' or pd.api.types.is_integer_dtype(df[col]):
                unique_categories = df[col].dropna().unique()
                categories.append(unique_categories)
                df[col] = pd.Categorical(df[col], categories=unique_categories)
                df[col] = df[col].cat.codes

    # 處理遺漏值
    handle_missing_values(df)

    # 正規化數值型特徵
    scaler = StandardScaler()
    df[df.select_dtypes(include=['float']).columns] = scaler.fit_transform(df.select_dtypes(include=['float']))

    # 從 DataFrame 轉換為 numpy arrays
    cat = df.select_dtypes(include=['int', 'category']).values
    flt = df.select_dtypes(include=['float']).values

    # 每個分類特徵的唯一值數量
    num_cat = (cat.max(axis=0) + 1).tolist() if cat.size > 0 else []

    if labelled:
        return lab, cat, flt, num_cat, categories
    else:
        return cat, flt, num_cat

# 定義資料集
class PPD(torch.utils.data.Dataset):
    def __init__(self, cat_data, num_data, label=None):
        self.cat_data = cat_data
        self.num_data = num_data
        self.label = label

    def t(self, x):
        return torch.from_numpy(x)

    def __getitem__(self, idx):
        if self.label is not None:
            return self.t(self.cat_data[idx]).long(), self.t(self.num_data[idx]).float(), torch.tensor(self.label[idx]).float()
        else:
            return self.t(self.cat_data[idx]).long(), self.t(self.num_data[idx]).float()

    def __len__(self):
        return len(self.cat_data)

train_label, train_cat_data, train_num_data, num_cat, categories = load_csv(data_dir + '/train.csv', labelled=True)
train_dataset, valid_dataset = torch.utils.data.random_split(PPD(train_cat_data, train_num_data, label=train_label), [int(0.8 * len(train_cat_data)), len(train_cat_data) - int(0.8 * len(train_cat_data))])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

class ImprovedNN(nn.Module):
    def __init__(self, num_cat, emb_size=96):
        super().__init__()
        self.num_cat = num_cat
        if len(num_cat) > 0:
            self.emb = nn.ModuleList([nn.Embedding(n, emb_size) for n in num_cat])
        else:
            self.emb = None
        number_of_emb = emb_size * len(num_cat) + train_num_data.shape[1]

        self.fc = nn.Sequential(
            nn.Linear(number_of_emb, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x_cat, x_num):
        if self.emb is not None and x_cat.size(1) > 0:
            x_cat_emb = [self.emb[i](x_cat[:, i]) for i in range(x_cat.size(1))]
            x_cat_emb = torch.cat(x_cat_emb, dim=1) if len(x_cat_emb) > 0 else torch.zeros((x_cat.size(0), 0), device=x_num.device)
            x = torch.cat([x_cat_emb, x_num], dim=1)
        else:
            x = x_num
        x = self.fc(x)
        return x.squeeze()

model = ImprovedNN(num_cat)

# 定義替代損失
class AUROCSurrogateLoss(nn.Module):
    def __init__(self):
        super(AUROCSurrogateLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        pos_pairs = (y_true == 1).nonzero(as_tuple=True)[0]
        neg_pairs = (y_true == 0).nonzero(as_tuple=True)[0]
        
        if len(pos_pairs) == 0 or len(neg_pairs) == 0:
            return torch.tensor(0.0, requires_grad=True)

        loss = 0
        for p in pos_pairs:
            for n in neg_pairs:
                loss += torch.relu(1 - (y_pred[p] - y_pred[n]))
        
        return loss / (len(pos_pairs) * len(neg_pairs))

loss_function = AUROCSurrogateLoss()

# 使用 AUROC 類計算 AUC
auc_metric = AUROC(task='binary', num_classes=1)

# 訓練函數
def train(model, train_loader, valid_loader, config):
    model.to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_auc = 0
    patience = 5
    no_improve_epochs = 0

    for epoch in range(config.num_epochs):
        model.train()
        all_y_true, all_y_pred, all_loss = [], [], []
        for x_cat, x_num, y in tqdm(train_loader):
            x_cat, x_num, y = x_cat.to(config.device), x_num.to(config.device), y.to(config.device).float()
            y_pred = model(x_cat, x_num)

            loss = loss_function(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_y_true.append(y)
            all_y_pred.append(y_pred if y_pred.dim() > 0 else y_pred.unsqueeze(0))
            all_loss.append(loss)
        
        all_y_true = torch.cat(all_y_true)
        all_y_pred = torch.cat(all_y_pred)
        auc_score = auc_metric(torch.sigmoid(all_y_pred), all_y_true.int()).item()
        loss = torch.mean(torch.stack(all_loss))
        print(f'Epoch {epoch+1}/{config.num_epochs}, train loss: {loss.item()}, train AUC: {auc_score}')

        model.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []
            for x_cat, x_num, y in valid_loader:
                x_cat, x_num, y = x_cat.to(config.device), x_num.to(config.device), y.to(config.device).float()
                y_true.append(y)
                y_pred.append(model(x_cat, x_num) if model(x_cat, x_num).dim() > 0 else model(x_cat, x_num).unsqueeze(0))
            y_true = torch.cat(y_true)
            y_pred = torch.cat(y_pred)
            auc_score = auc_metric(torch.sigmoid(y_pred), y_true.int()).item()
            loss = loss_function(y_pred, y_true)
            print(f'Epoch {epoch+1}/{config.num_epochs}, valid loss: {loss.item()}, valid AUC: {auc_score}')

            if auc_score > best_auc:
                best_auc = auc_score
                no_improve_epochs = 0
                torch.save(model.state_dict(), config.ckpt_path)
                print(f'==== best valid AUC: {auc_score} ====')
            else:
                no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

        scheduler.step(auc_score)
    model.load_state_dict(torch.load(config.ckpt_path))

train(model, train_loader, valid_loader, config)

# 測試集預測
test_cat_data, test_num_data, nc = load_csv(data_dir + '/test.csv', labelled=False, categories=categories)
test_dataset = PPD(test_cat_data, test_num_data)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

predictions = []
with torch.no_grad():
    model.eval()
    for x_cat, x_num in test_loader:
        x_cat, x_num = x_cat.to(config.device), x_num.to(config.device)
        predictions.extend(torch.sigmoid(model(x_cat, x_num)).cpu().tolist())

# 保存預測結果
with open('predict.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'label'])
    for i, p in enumerate(predictions):
        writer.writerow([i, p])

# Feature analysis
categorical_features = ['urbanization', 'Sex_boy', 'unexpectedPre', 'firstborn', 'singleton',
        'edu_M_high', 'martial', 'Somke_perinatal', 'family support_good',
        'income', 'baby_health', 'Alcohol_preg', 'Jobstress_preg',
        'Job_stress_6m', 'AD_B', 'AS_M', 'AR_M', 'AD_M', 'allergy_M', 'PET',
        'incense', 'mold', 'water_wall', 'Vitamin_tri1', 'Vitamin_tri23',
        'Vitamin_p', 'fishliveroil_tri1', 'fishliveroil_tri23', 'fishoil_tri1',
        'fishoil_tri23', 'antibiotic_p', 'Vitamin_post', 'fishliveroil_post',
        'Probiotics_post', 'burn_around_house', 'industry_around_house',
        'odor_around_house', 'paintrenov_p', 'paintrenov_post',
        'pure_day_time_work_preg', 'pure_night_work_preg', 'shift_work_preg',
        'pure_day_time_work_6m', 'pure_night_work_6m', 'shift_work_6m']
continuous_features = ['birth_month', 'Gestation', 'birth_weight', 'Parity', 'multibirth',
        'mother_age', 'Height_M', 'BW_before_P', 'BMI_before_P', 'BW_before_D',
        'BWgain', 'CS', 'tocolysis', 'breastfeed', 'APGAR_1', 'APGAR_5',
        'edu_F_high', 'Somke_perg.', 'sencondsmoke_6m', 'Y0_CO', 'Tri1_CO',
        'Tri2_CO', 'Tri3_CO', '365_CO', 'Y0_NO2', 'Tri1_NO2', 'Tri2_NO2',
        'Tri3_NO2', '365_NO2', 'Y0_PM25', 'Tri1_PM25', 'Tri2_PM25', 'Tri3_PM25',
        '365_PM25', 'Tri1_PM10', 'Tri2_PM10', 'Tri3_PM10', 'Tri1_SO2',
        'Tri2_SO2', 'Tri3_SO2', 'Tri1_O3', 'Tri2_O3', 'Tri3_O3', 'Tri1_NOx',
        'Tri2_NOx', 'Tri3_NOx', 'Tri1_NO', 'Tri2_NO', 'Tri3_NO', 'Tri1_TEMP',
        'Tri2_TEMP', 'Tri3_TEMP', 'Mean NOx', 'Mean NO', 'Mean TEMP', 'Tri1_RH',
        'Tri2_RH', 'Tri3_RH']        
all_auc = []
model.eval()

with torch.no_grad():
    for i in range(len(num_cat)):
        auc_ = 0
        for x_cat, x_num, y in valid_loader:
            x_cat, x_num, y = x_cat.to(config.device), x_num.to(config.device), y.to(config.device)
            x_cat[:, i] = 0  # Remove contribution of this category
            y_pred = model(x_cat, x_num)
            auc_ += auc_metric(torch.sigmoid(y_pred), y.int()).item()
        all_auc.append(auc_ / len(valid_loader))

    for i in range(len(num_cat), len(num_cat) + train_num_data.shape[1]):
        auc_ = 0
        for x_cat, x_num, y in valid_loader:
            x_cat, x_num, y = x_cat.to(config.device), x_num.to(config.device), y.to(config.device)
            feature_idx = i - len(num_cat)
            if feature_idx < x_num.size(1):
                x_num[:, feature_idx] = -100  # Mask out contribution of this feature
            y_pred = model(x_cat, x_num)
            auc_ += auc_metric(torch.sigmoid(y_pred), y.int()).item()
        all_auc.append(auc_ / len(valid_loader))

    normal_auc = 0
    for x_cat, x_num, y in valid_loader:
        x_cat, x_num, y = x_cat.to(config.device), x_num.to(config.device), y.to(config.device)
        y_pred = model(x_cat, x_num)
        normal_auc += auc_metric(torch.sigmoid(y_pred), y.int()).item()
    normal_auc /= len(valid_loader)

    delta = np.array(all_auc) - normal_auc

# Print top 5 categorical and numerical features contributing to the AUC
idx_cat = np.argsort(delta[:len(num_cat)])
idx_num = np.argsort(delta[len(num_cat):])
print("Top 5 categorical features affecting AUC:", idx_cat[:5])
print("Top 5 numerical features affecting AUC:", idx_num[:5])

 
# Plot the results using plt.bar
plt.figure()
plt.bar(np.arange(len(idx_cat)), delta[idx_cat], tick_label=np.array(categorical_features)[idx_cat])
plt.xlabel("Categorical Feature Name")
plt.ylabel("Delta AUC")
plt.title("Top Categorical Features Affecting AUC")
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability


plt.figure()
plt.bar(np.arange(train_num_data.shape[1]), delta[idx_num + len(num_cat)])
plt.xlabel("Numerical Feature Index")
plt.ylabel("Delta AUC")
plt.title("Top Numerical Features Affecting AUC")

plt.show()

# Calculate the gradient of the loss with respect to each feature
# For categorical features, calculate the gradient of the loss with respect to the embedding, and average over embed dimensions
model.eval()
grads_num = []

for x_cat, x_num, y in valid_loader:
    x_cat, x_num, y = x_cat.to(config.device), x_num.to(config.device), y.to(config.device)
    x_num.requires_grad = True
    y_pred = model(x_cat, x_num)
    loss = loss_function(y_pred, y)
    loss.backward()
    grads_num.append(x_num.grad.abs())
grads_cat = np.array([model.emb[i].weight.grad.abs().mean().cpu() for i in range(len(num_cat))])
grads_num = torch.cat(grads_num, dim=0).mean(dim=0).cpu().numpy()

i_cat = np.argsort(grads_cat)[::-1]
i_num = np.argsort(grads_num)[::-1]

print(i_cat[:5])
print(i_num[:5])

# plot the gradients for the categorical features
plt.figure()
plt.bar(range(len(num_cat)), grads_cat[i_cat])
plt.figure()
plt.bar(range(train_num_data.shape[1]), grads_num[i_num])