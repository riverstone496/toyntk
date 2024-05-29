import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.func import vmap, jacrev
import numpy as np

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed=1233
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# CIFAR-10データセットのダウンロードとロード
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# クラス0と1のみをフィルタリングする関数
def filter_01(dataset):
    mask = (torch.tensor(dataset.targets) == 8) | (torch.tensor(dataset.targets) == 9)
    dataset.data = dataset.data[mask.numpy()]
    dataset.targets = torch.tensor(dataset.targets)[mask].numpy()
    return dataset

train_set = filter_01(train_set)
test_set = filter_01(test_set)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

# トレーニングデータの一部を取得
x_train, y_train = next(iter(train_loader))

# デバイスに転送
x_train, y_train = x_train.to(device), y_train.to(device)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 1024)  # 32x32x3に変更
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2)  # クラス数を2に変更

    def forward(self, x):
        x = x.view(-1, 32*32*3)  # 32x32x3に変更
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN().to(device)

def fnet_single(params, x):
    x = x.view(-1, 32*32*3)  # 32x32x3に変更
    x = F.relu(F.linear(x, params[0], params[1]))
    x = F.relu(F.linear(x, params[2], params[3]))
    x = F.linear(x, params[4], params[5])
    return x

params = [p for p in model.parameters()]

def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2, block_size=32):
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1_flat = torch.cat([j.reshape(j.shape[0], -1) for j in jac1], dim=1)
    
    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2_flat = torch.cat([j.reshape(j.shape[0], -1) for j in jac2], dim=1)

    # Initialize result matrix
    num_x1 = jac1_flat.shape[0]
    num_x2 = jac2_flat.shape[0]
    result = torch.zeros((num_x1, num_x2), device=jac1_flat.device)

    # Block-wise computation to save memory
    for i in range(0, num_x1, block_size):
        for j in range(0, num_x2, block_size):
            block_jac1 = jac1_flat[i:i+block_size]
            block_jac2 = jac2_flat[j:j+block_size]
            result[i:i+block_size, j:j+block_size] = block_jac1 @ block_jac2.T
    return result

print('empirical_ntk_jacobian_contraction')
ntk = empirical_ntk_jacobian_contraction(fnet_single, params, x_train, x_train)

def predict_with_ntk(ntk, x_train, y_train, x_test, reg=1e-4):
    # Compute Gram matrix and its inverse
    K_train_train = ntk[:len(x_train), :len(x_train)]
    K_train_train += reg * torch.eye(K_train_train.shape[-1], device=device)  # Regularization
    K_train_train_inv = torch.inverse(K_train_train)
    alpha = K_train_train_inv @ y_train.float()

    preds_list = []
    for i in range(0, len(x_test), 128):
        x_test_batch = x_test[i:i+128]
        K_train_test = empirical_ntk_jacobian_contraction(fnet_single, params, x_train, x_test_batch)
        preds_batch = (K_train_test.transpose(0, 1) @ alpha).argmax(dim=1)
        preds_list.append(preds_batch)
    
    return torch.cat(preds_list, dim=0)

# モデルを使用して予測
y_train_one_hot = F.one_hot(y_train-8, num_classes=2).float()  # クラス数を2に変更
print('predict_with_ntk')

# テストデータをバッチごとに処理
test_preds_list = []
test_targets_list = []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)-8
        preds = predict_with_ntk(ntk, x_train, y_train_one_hot, x_batch)
        test_preds_list.append(preds)
        test_targets_list.append(y_batch)

test_preds = torch.cat(test_preds_list, dim=0)
test_targets = torch.cat(test_targets_list, dim=0)

# 精度の計算
accuracy = (test_preds == test_targets).float().mean().item()
print(f'Test Accuracy: {accuracy * 100:.2f}%')
