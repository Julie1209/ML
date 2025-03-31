# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 02:25:12 2024

@author: cdpss
"""
#%%HW3Question1
import numpy as np
from sklearn.decomposition import PCA

X = np.array([
    [1, 2, 3],
    [4, 8, 5],
    [3, 12, 9],
    [1, 8, 5],
    [5, 14, 2],
    [7, 4, 1],
    [9, 8, 9],
    [3, 8, 1],
    [11, 5, 6],
    [10, 11, 7]
])

mean_X = np.mean(X, axis=0)
X_centered = X - mean_X

cov_matrix = np.cov(X_centered, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

principal_components = np.dot(X_centered, eigenvectors)

principal_components_2D = principal_components[:, :2]

X_reconstructed = np.dot(principal_components_2D, eigenvectors[:, :2].T) + mean_X

reconstruction_errors = np.sum((X - X_reconstructed) ** 2, axis=1)
average_reconstruction_error = np.mean(reconstruction_errors)

print("特徵值：\n", eigenvalues)
print("\n特徵向量：\n", eigenvectors)
print("\n每個樣本的主成分：\n", principal_components)
print("\n平均重構誤差：", average_reconstruction_error)
#%%HW3Question3-3
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# 定義之前計算出的拉普拉斯矩陣 L
L = np.array([
    [3, -1, -1, -1, 0, 0, 0, 0, 0, 0],
    [-1, 3, 0, -1, 0, 0, 0, -1, 0, 0],
    [-1, 0, 2, 0, 0, 0, 0, -1, 0, 0],
    [-1, -1, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, -1, -1, 0, 0, 0],
    [0, 0, 0, 0, -1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 2, 0, 0, -1],
    [0, -1, -1, 0, 0, 0, 0, 3, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, -1, 2, -1],
    [0, 0, 0, 0, 0, 0, -1, 0, -1, 2]
])

# 定義對角矩陣 D
D = np.diag([3, 3, 2, 2, 2, 1, 2, 3, 2, 2])

# 解廣義特徵值問題 L Ψ = λ D Ψ
eigenvalues, eigenvectors = eigh(L, D)

# 選取對應於最小非零特徵值的三個特徵向量
Psi = eigenvectors[:, 1:4]  # 取第 2、3、4 小的特徵值對應的特徵向量

# 在3D空間中繪製節點的嵌入結果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 繪製 10 個節點在 3D 空間中的坐標點
ax.scatter(Psi[:, 0], Psi[:, 1], Psi[:, 2], c='b', marker='o')

# 添加節點標籤
for i in range(10):
    if i == 0:  # Node 1 特製
        ax.text(Psi[i, 0] + 0.02, Psi[i, 1] + 0.02, Psi[i, 2] - 0.05, f'Node {i+1}', color='black', fontsize=7)
    else:
        ax.text(Psi[i, 0] + 0.02, Psi[i, 1] + 0.02, Psi[i, 2] - 0.02, f'Node {i+1}', color='black', fontsize=7)

# 顯示結果
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Scatter Plot of Reduced Points')
plt.show()

#%%HW3Question3-4
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Define Laplacian matrix L
L = np.array([
    [3, -1, -1, -1, 0, 0, 0, 0, 0, 0],
    [-1, 3, 0, -1, 0, 0, 0, -1, 0, 0],
    [-1, 0, 2, 0, 0, 0, 0, -1, 0, 0],
    [-1, -1, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, -1, -1, 0, 0, 0],
    [0, 0, 0, 0, -1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 2, 0, 0, -1],
    [0, -1, -1, 0, 0, 0, 0, 3, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, -1, 2, -1],
    [0, 0, 0, 0, 0, 0, -1, 0, -1, 2]
])

# Define diagonal matrix D
D = np.diag([3, 3, 2, 2, 2, 1, 2, 3, 2, 2])

# Solve the generalized eigenvalue problem L Ψ = λ D Ψ
eigenvalues, eigenvectors = eigh(L, D)

# Print eigenvalues
print("Eigenvalues:")
print(eigenvalues)

# Select the second, third, and fourth smallest eigenvalues
Psi = eigenvectors[:, 1:4]  # Columns 1 to 3

# Verify Trace(Ψᵗ L Ψ) = 1.098
trace_value = np.trace(Psi.T @ L @ Psi)
print(f"\nTrace(Ψᵗ L Ψ) = {trace_value:.3f}")

# Verify Ψᵗ D Ψ = I_3
psi_d_psi = Psi.T @ D @ Psi
print("\nΨᵗ D Ψ =")
print(psi_d_psi)

# Plot the reduced points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 10 nodes in 3D space
ax.scatter(Psi[:, 0], Psi[:, 1], Psi[:, 2], c='b', marker='o')

# Add node labels
for i in range(10):
    ax.text(Psi[i, 0], Psi[i, 1], Psi[i, 2], f'Node {i+1}', fontsize=9)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Scatter Plot of Reduced Points')
plt.show()