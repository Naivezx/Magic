import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau

FACE_SIZE = 54
NUM_CLASSES = 18
MOVE_LABELS = ['U', 'U\'', 'U2', 'D', 'D\'', 'D2', 'L', 'L\'', 'L2', 'R', 'R\'', 'R2', 'F', 'F\'', 'F2', 'B', 'B\'', 'B2']
MOVE_TO_IDX = {m: i for i, m in enumerate(MOVE_LABELS)}
IDX_TO_MOVE = {i: m for m, i in MOVE_TO_IDX.items()}

class CubeDataset(Dataset):
    def __init__(self, folder):
        self.samples = []
        for fname in os.listdir(folder):
            if not fname.startswith("training.seq."):
                continue
            path = os.path.join(folder, fname)
            with open(path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                i = 0
                while i < len(lines):
                    try:
                        state = list(map(int, lines[i].split()))
                        if len(state) != 54:
                            print(f"Invalid state length at {path}, line {i+1}: {lines[i]}")
                            i += 1
                            continue
                    except ValueError:
                        print(f"Error parsing state at {path}, line {i+1}: {lines[i]}")
                        i += 1
                        continue
                    i += 1
                    if i >= len(lines):
                        break
                    move = lines[i]
                    if move == '#':
                        i += 1
                        continue
                    if move not in MOVE_TO_IDX:
                        print(f"Invalid move at {path}, line {i+1}: {move}")
                        i += 1
                        continue
                    self.samples.append((torch.tensor(state, dtype=torch.float32), MOVE_TO_IDX[move]))
                    i += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return x.view(6, 3, 3), y
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 殞差連接的調整層（若輸入輸出通道數不同）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)  # 殞差連接
        out = self.relu(out)
        return out
    
class CubeSolverCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 添加殞差塊
        self.res_blocks = nn.Sequential(
            ResidualBlock(64, 64, stride=1),   # 第一個殞差塊，保持 64 通道
            ResidualBlock(64, 128, stride=1),  # 第二個殞差塊，增加到 128 通道
            ResidualBlock(128, 256, stride=1)  # 第三個殞差塊，增加到 256 通道
        )
        
        self.flatten = nn.Flatten()
        
        # MLP 部分，調整輸入維度以匹配殞差塊輸出
        self.mlp = nn.Sequential(
            nn.Linear(256 * 3 * 3, 512),  # 假設輸入為 3x3，根據實際輸出調整
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )
    
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x

def train_model_simple(num_epochs=10, val_ratio=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = CubeDataset('seq')
    
    # 計算動作權重
    action_counts = Counter(action for _, action in dataset.samples)
    total_samples = len(dataset)
    weights = torch.tensor([total_samples / (len(MOVE_LABELS) * action_counts.get(i, 1)) for i in range(len(MOVE_LABELS))], dtype=torch.float32).to(device)
    
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_ratio, random_state=42, shuffle=True)
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=128)
    model = CubeSolverCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights) 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss, correct_train, total_train = 0, 0, 0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for x, y in train_loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            correct_train += (out.argmax(dim=1) == y).sum().item()
            total_train += y.size(0)
        
        train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        model.eval()
        total_val_loss, correct_val, total_val = 0, 0, 0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for x, y in val_loop:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                total_val_loss += loss.item()
                correct_val += (out.argmax(dim=1) == y).sum().item()
                total_val += y.size(0)
        
        val_loss = total_val_loss / len(val_loader)
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        scheduler.step(val_loss) 

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'cube_model_epoch_{epoch+1}.pt')
            print(f"Saved model: cube_model_epoch_{epoch+1}.pt")
            plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epoch + 1, filename=f'training_metrics_epoch_{epoch+1}.png')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'cube_model.pt')
    
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs, filename='training_metrics_final.png')
    return model

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs, filename='training_metrics.png'):
    epochs = range(1, num_epochs + 1)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Training metrics plot saved as '{filename}'")
    plt.close()

if __name__ == '__main__':
    model = train_model_simple(num_epochs=100, val_ratio=0.1)


