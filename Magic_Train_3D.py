import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
import numpy as np
import keras_core as keras
import keras_core.layers as layers
import keras_core.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

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
                            print(f"無效的狀態長度於 {path}, 第 {i+1} 行: {lines[i]}")
                            i += 1
                            continue
                    except ValueError:
                        print(f"解析狀態錯誤於 {path}, 第 {i+1} 行: {lines[i]}")
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
                        print(f"無效的動作於 {path}, 第 {i+1} 行: {move}")
                        i += 1
                        continue
                    self.samples.append((torch.tensor(state, dtype=torch.float32), MOVE_TO_IDX[move]))
                    i += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        # 將數據重塑為 (1, 6, 3, 3) 以用於 3D 卷積
        x = x.view(1, 6, 3, 3)
        return x, y

def residual_block(x, filters, stride=1):
    shortcut = x
    x = layers.Conv3D(filters, kernel_size=(3, 3, 3), strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv3D(filters, kernel_size=(3, 3, 3), strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv3D(filters, kernel_size=(1, 1, 1), strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def create_cube_solver_model():
    inputs = keras.Input(shape=(1, 6, 3, 3))
    x = layers.Conv3D(64, kernel_size=(3, 3, 3), strides=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 128, stride=1)
    x = residual_block(x, 256, stride=1)
    
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
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

class CustomReduceLROnPlateau(keras.callbacks.Callback):
    def __init__(self, factor=0.5, patience=10, min_lr=1e-6, verbose=1):
        super().__init__()
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_loss = float('inf')  
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if current_loss is None:
            return

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_weights = self.model.get_weights()
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                current_lr = float(self.model.optimizer.learning_rate)
                new_lr = max(current_lr * self.factor, self.min_lr)
                if new_lr < current_lr:
                    self.model.optimizer.learning_rate.assign(new_lr)
                    if self.verbose:
                        print(f"\nEpoch {epoch + 1}: 將學習率降低至 {new_lr:.6f}")
                self.wait = 0

def train_model_simple(num_epochs=100, val_ratio=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = CubeDataset('seq')
    action_counts = Counter(action for _, action in dataset.samples)
    total_samples = len(dataset)
    class_weights = torch.tensor([total_samples / (len(MOVE_LABELS) * action_counts.get(i, 1)) for i in range(len(MOVE_LABELS))], dtype=torch.float32).to(device)
    
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_ratio, random_state=42, shuffle=True)
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=128)
    
    model = create_cube_solver_model()
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    lr_scheduler = CustomReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        total_train_loss, correct_train, total_train = 0, 0, 0
        train_loop = tqdm(train_loader, desc="訓練")
        for x, y in train_loop:
            x, y = x.to(device), y.to(device)
            x_np, y_np = x.cpu().numpy(), y.cpu().numpy()
            history = model.fit(x_np, y_np, batch_size=x.shape[0], epochs=1, verbose=0, 
                              class_weight={i: class_weights[i].item() for i in range(NUM_CLASSES)},
                              callbacks=[lr_scheduler])
            total_train_loss += history.history['loss'][0] * x.shape[0]
            correct_train += history.history['accuracy'][0] * x.shape[0]
            total_train += x.shape[0]
        
        train_loss = total_train_loss / len(train_subset)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        model.eval()
        total_val_loss, correct_val, total_val = 0, 0, 0
        val_loop = tqdm(val_loader, desc="驗證")
        with torch.no_grad():
            for x, y in val_loop:
                x, y = x.to(device), y.to(device)
                x_np, y_np = x.cpu().numpy(), y.cpu().numpy()
                history = model.evaluate(x_np, y_np, batch_size=x.shape[0], verbose=0, return_dict=True)
                total_val_loss += history['loss'] * x.shape[0]
                correct_val += history['accuracy'] * x.shape[0]
                total_val += x.shape[0]
        
        val_loss = total_val_loss / len(val_subset)
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}, 訓練損失: {train_loss:.4f}, 訓練準確率: {train_acc:.4f}, 驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.4f}")
        
        lr_scheduler.on_epoch_end(epoch, logs={'val_loss': val_loss})
        if (epoch + 1) % 5 == 0:
            model.save_weights(f'cube_model_epoch_{epoch+1}.weights.h5')
            print(f"已保存模型權重: cube_model_epoch_{epoch+1}.weights.h5")
            plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epoch + 1, filename=f'training_metrics_epoch_{epoch+1}.png')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights('cube_model.weights.h5')
    
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, num_epochs, filename='training_metrics_final.png')
    return model

if __name__ == '__main__':
    model = train_model_simple(num_epochs=100, val_ratio=0.1)