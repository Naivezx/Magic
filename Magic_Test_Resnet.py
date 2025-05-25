import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from scipy.stats import wasserstein_distance
import argparse
from heapq import heappush 
# ----------------------
# Constants
# ----------------------
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

# ----------------------
# Accuracy Evaluation
# ----------------------
def evaluate_model(model):
    device = next(model.parameters()).device
    dataset = CubeDataset('seq')
    loader = DataLoader(dataset, batch_size=128)
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"Accuracy: {correct / total * 100:.2f}%")

# ----------------------
# Diffusion Distance
# ----------------------
def diffusion_distance(state):
    solved_state = [i//9 for i in range(54)]
    state_dist = []
    solved_dist = []
    for face in range(6):
        state_face = state[face*9:(face+1)*9]
        solved_face = solved_state[face*9:(face+1)*9]
        state_counts = np.bincount(state_face, minlength=6)
        solved_counts = np.bincount(solved_face, minlength=6)
        state_dist.append(state_counts / 9.0)
        solved_dist.append(solved_counts / 9.0)
    total_distance = 0
    for s_dist, sol_dist in zip(state_dist, solved_dist):
        total_distance += wasserstein_distance(np.arange(6), np.arange(6), s_dist, sol_dist)
    return total_distance

# ----------------------
# Inference
# ----------------------
def predict_move(model, state):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = torch.tensor(state, dtype=torch.float32).view(1, 6, 3, 3).to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
        return IDX_TO_MOVE[pred]

# ----------------------
# Helper Functions for State Update
# ----------------------
def rotate_face(face, direction='clockwise'):
    if direction == 'clockwise':
        return [face[6], face[3], face[0], face[7], face[4], face[1], face[8], face[5], face[2]]
    else:
        return [face[2], face[5], face[8], face[1], face[4], face[7], face[0], face[3], face[6]]

def apply_move(state, move):
    new_state = state.copy()
    U, D, L, R, F, B = range(0, 54, 9)
    
    if move == 'U':
        new_state[U:U+9] = rotate_face(state[U:U+9], 'clockwise')
        new_state[F:F+3], new_state[R:R+3], new_state[B:B+3], new_state[L:L+3] = \
            state[L:L+3], state[F:F+3], state[R:R+3], state[B:B+3]
    elif move == 'U\'':
        new_state[U:U+9] = rotate_face(state[U:U+9], 'counterclockwise')
        new_state[L:L+3], new_state[B:B+3], new_state[R:R+3], new_state[F:F+3] = \
            state[F:F+3], state[R:R+3], state[B:B+3], state[L:L+3]
    elif move == 'U2':
        new_state[U:U+9] = rotate_face(rotate_face(state[U:U+9], 'clockwise'), 'clockwise')
        new_state[F:F+3], new_state[L:L+3], new_state[R:R+3], new_state[B:B+3] = \
            state[R:R+3], state[B:B+3], state[L:L+3], state[F:F+3]
    elif move == 'D':
        new_state[D:D+9] = rotate_face(state[D:D+9], 'clockwise')
        new_state[F+6:F+9], new_state[L+6:L+9], new_state[R+6:R+9], new_state[B+6:B+9] = \
            state[R+6:R+9], state[F+6:F+9], state[B+6:B+9], state[L+6:L+9]
    elif move == 'D\'':
        new_state[D:D+9] = rotate_face(state[D:D+9], 'counterclockwise')
        new_state[R+6:R+9], new_state[F+6:F+9], new_state[B+6:B+9], new_state[L+6:L+9] = \
            state[F+6:F+9], state[L+6:L+9], state[R+6:R+9], state[B+6:B+9]
    elif move == 'D2':
        new_state[D:D+9] = rotate_face(rotate_face(state[D:D+9], 'clockwise'), 'clockwise')
        new_state[F+6:F+9], new_state[R+6:R+9], new_state[L+6:L+9], new_state[B+6:B+9] = \
            state[L+6:L+9], state[B+6:B+9], state[F+6:F+9], state[R+6:R+9]
    elif move == 'L':
        new_state[L:L+9] = rotate_face(state[L:L+9], 'clockwise')
        temp = [state[U+0], state[U+3], state[U+6]]
        new_state[U+0], new_state[U+3], new_state[U+6] = state[B+8], state[B+5], state[B+2]
        new_state[B+8], new_state[B+5], new_state[B+2] = state[D+0], state[D+3], state[D+6]
        new_state[D+0], new_state[D+3], new_state[D+6] = state[F+0], state[F+3], state[F+6]
        new_state[F+0], new_state[F+3], new_state[F+6] = temp
    elif move == 'L\'':
        new_state[L:L+9] = rotate_face(state[L:L+9], 'counterclockwise')
        temp = [state[U+0], state[U+3], state[U+6]]
        new_state[U+0], new_state[U+3], new_state[U+6] = state[F+0], state[F+3], state[F+6]
        new_state[F+0], new_state[F+3], new_state[F+6] = state[D+0], state[D+3], state[D+6]
        new_state[D+0], new_state[D+3], new_state[D+6] = state[B+8], state[B+5], state[B+2]
        new_state[B+8], new_state[B+5], new_state[B+2] = temp
    elif move == 'L2':
        new_state[L:L+9] = rotate_face(rotate_face(state[L:L+9], 'clockwise'), 'clockwise')
        for _ in range(2):
            temp = [new_state[U+0], new_state[U+3], new_state[U+6]]
            new_state[U+0], new_state[U+3], new_state[U+6] = new_state[B+8], new_state[B+5], new_state[B+2]
            new_state[B+8], new_state[B+5], new_state[B+2] = new_state[D+0], new_state[D+3], new_state[D+6]
            new_state[D+0], new_state[D+3], new_state[D+6] = new_state[F+0], new_state[F+3], new_state[F+6]
            new_state[F+0], new_state[F+3], new_state[F+6] = temp
    elif move == 'R':
        new_state[R:R+9] = rotate_face(state[R:R+9], 'clockwise')
        temp = [state[U+2], state[U+5], state[U+8]]
        new_state[U+2], new_state[U+5], new_state[U+8] = state[F+2], state[F+5], state[F+8]
        new_state[F+2], new_state[F+5], new_state[F+8] = state[D+2], state[D+5], state[D+8]
        new_state[D+2], new_state[D+5], new_state[D+8] = state[B+6], state[B+3], state[B+0]
        new_state[B+6], new_state[B+3], new_state[B+0] = temp
    elif move == 'R\'':
        new_state[R:R+9] = rotate_face(state[R:R+9], 'counterclockwise')
        temp = [state[U+2], state[U+5], state[U+8]]
        new_state[U+2], new_state[U+5], new_state[U+8] = state[B+6], state[B+3], state[B+0]
        new_state[B+6], new_state[B+3], new_state[B+0] = state[D+2], state[D+5], state[D+8]
        new_state[D+2], new_state[D+5], new_state[D+8] = state[F+2], state[F+5], state[F+8]
        new_state[F+2], new_state[F+5], new_state[F+8] = temp
    elif move == 'R2':
        new_state[R:R+9] = rotate_face(rotate_face(state[R:R+9], 'clockwise'), 'clockwise')
        for _ in range(2):
            temp = [new_state[U+2], new_state[U+5], new_state[U+8]]
            new_state[U+2], new_state[U+5], new_state[U+8] = new_state[F+2], new_state[F+5], new_state[F+8]
            new_state[F+2], new_state[F+5], new_state[F+8] = new_state[D+2], new_state[D+5], new_state[D+8]
            new_state[D+2], new_state[D+5], new_state[D+8] = new_state[B+6], new_state[B+3], new_state[B+0]
            new_state[B+6], new_state[B+3], new_state[B+0] = temp
    elif move == 'F':
        new_state[F:F+9] = rotate_face(state[F:F+9], 'clockwise')
        temp = [state[U+6], state[U+7], state[U+8]]
        new_state[U+6], new_state[U+7], new_state[U+8] = state[L+8], state[L+5], state[L+2]
        new_state[L+8], new_state[L+5], new_state[L+2] = state[D+2], state[D+1], state[D+0]
        new_state[D+2], new_state[D+1], new_state[D+0] = state[R+0], state[R+3], state[R+6]
        new_state[R+0], new_state[R+3], new_state[R+6] = temp
    elif move == 'F\'':
        new_state[F:F+9] = rotate_face(state[F:F+9], 'counterclockwise')
        temp = [state[U+6], state[U+7], state[U+8]]
        new_state[U+6], new_state[U+7], new_state[U+8] = state[R+0], state[R+3], state[R+6]
        new_state[R+0], new_state[R+3], new_state[R+6] = state[D+2], state[D+1], state[D+0]
        new_state[D+2], new_state[D+1], new_state[D+0] = state[L+8], state[L+5], state[L+2]
        new_state[L+8], new_state[L+5], new_state[L+2] = temp
    elif move == 'F2':
        new_state[F:F+9] = rotate_face(rotate_face(state[F:F+9], 'clockwise'), 'clockwise')
        for _ in range(2):
            temp = [new_state[U+6], new_state[U+7], new_state[U+8]]
            new_state[U+6], new_state[U+7], new_state[U+8] = new_state[L+8], new_state[L+5], new_state[L+2]
            new_state[L+8], new_state[L+5], new_state[L+2] = new_state[D+2], new_state[D+1], new_state[D+0]
            new_state[D+2], new_state[D+1], new_state[D+0] = new_state[R+0], new_state[R+3], new_state[R+6]
            new_state[R+0], new_state[R+3], new_state[R+6] = temp
    elif move == 'B':
        new_state[B:B+9] = rotate_face(state[B:B+9], 'clockwise')
        temp = [state[U+0], state[U+1], state[U+2]]
        new_state[U+0], new_state[U+1], new_state[U+2] = state[R+2], state[R+5], state[R+8]
        new_state[R+2], new_state[R+5], new_state[R+8] = state[D+8], state[D+7], state[D+6]
        new_state[D+8], new_state[D+7], new_state[D+6] = state[L+6], state[L+3], state[L+0]
        new_state[L+6], new_state[L+3], new_state[L+0] = temp
    elif move == 'B\'':
        new_state[B:B+9] = rotate_face(state[B:B+9], 'counterclockwise')
        temp = [state[U+0], state[U+1], state[U+2]]
        new_state[U+0], new_state[U+1], new_state[U+2] = state[L+6], state[L+3], state[L+0]
        new_state[L+6], new_state[L+3], new_state[L+0] = state[D+8], state[D+7], state[D+6]
        new_state[D+8], new_state[D+7], new_state[D+6] = state[R+2], state[R+5], state[R+8]
        new_state[R+2], new_state[R+5], new_state[R+8] = temp
    elif move == 'B2':
        new_state[B:B+9] = rotate_face(rotate_face(state[B:B+9], 'clockwise'), 'clockwise')
        for _ in range(2):
            temp = [new_state[U+0], new_state[U+1], new_state[U+2]]
            new_state[U+0], new_state[U+1], new_state[U+2] = new_state[R+2], new_state[R+5], new_state[R+8]
            new_state[R+2], new_state[R+5], new_state[R+8] = new_state[D+8], new_state[D+7], new_state[D+6]
            new_state[D+8], new_state[D+7], new_state[D+6] = new_state[L+6], new_state[L+3], new_state[L+0]
            new_state[L+6], new_state[L+3], new_state[L+0] = temp
    
    return new_state

def is_solved(state):
    for i in range(0, 54, 9):
        face = state[i:i+9]
        if not all(x == face[0] for x in face):
            return False
    return True

# ----------------------
# Beam Search with Diffusion Distance
# ----------------------
def simulate_full_solve_beam(model, state, max_steps=50, beam_width=3, alpha=0.5):
    print("Start solving...")
    device = next(model.parameters()).device
    initial_dist = diffusion_distance(state)
    beams = [(0.0, state.copy(), [])]
    
    for step in range(max_steps):
        new_beams = []
        for score, current_state, move_history in beams:
            x = torch.tensor(current_state, dtype=torch.float32).view(1, 6, 3, 3).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0]
            top_probs, top_actions = torch.topk(probs, beam_width)
            
            for p, a in zip(top_probs, top_actions):
                action = IDX_TO_MOVE[a.item()]
                new_state = apply_move(current_state, action)
                new_dist = diffusion_distance(new_state)
                
                dist_score = new_dist / (initial_dist + 1e-6)
                prob_score = -np.log(p.item() + 1e-6)
                new_score = alpha * prob_score + (1 - alpha) * dist_score
                
                if is_solved(new_state):
                    print("Solved!")
                    print(f"Solution sequence: {' '.join(move_history + [action])}")
                    print(f"Steps: {len(move_history) + 1}")
                    return move_history + [action]
                
                heappush(new_beams, (new_score, new_state, move_history + [action]))
        
        beams = new_beams[:beam_width]
        print(f"Step {step + 1}: Best sequence: {' '.join(beams[0][2])}, Distance: {diffusion_distance(beams[0][1]):.4f}")
        
        for _, _, history in beams:
            if len(history) >= 4 and history[-4:] == [history[-1]] * 4:
                print("Cycle detected in a beam, continuing with other beams...")
                beams = [b for b in beams if b[2][-4:] != [b[2][-1]] * 4]
                if not beams:
                    print("All beams in cycle, stopping.")
                    break
        
        if not beams:
            break
    
    print("Max steps reached. Could not solve fully.")
    if beams:
        _, _, best_history = beams[0]
        print(f"Best attempted sequence: {' '.join(best_history)}")
        print(f"Final distance: {diffusion_distance(beams[0][1]):.4f}")
        return best_history
    else:
        print("No valid sequences found.")
        return []

# ----------------------
# Main
# ----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Rubik's Cube Solver")
    parser.add_argument('--model_path', type=str, default='cube_model.pt',
                        help="Path to the trained model file")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CubeSolverCNN().to(device)
    
    # Load the trained model
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    except RuntimeError as e:
        print(f"Error loading model from {args.model_path}: {e}")
        exit(1)
    
    # Ensure model is in evaluation mode and check BatchNorm2d running stats
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            print(f"BatchNorm2d layer '{name}' training mode: {module.training}")
            if module.running_mean is None or module.running_var is None:
                print(f"Warning: BatchNorm2d layer '{name}' has uninitialized running stats. Consider re-training the model.")
    # Evaluate model accuracy
    # evaluate_model(model)
    
    # Test with a sample state
    
    sample_state = [2,0,5,2,0,5,3,5,1,0,3,0,3,1,5,0,3,4,3,5,5 ,1 ,2 ,2 ,4 ,4 ,0 ,2 ,1 ,2 ,4 ,3 ,4 ,4 ,0 ,4 ,1 ,0 ,1 ,3 ,4 ,1 ,1 ,2 ,3 ,5 ,4 ,3 ,2 ,5 ,0 ,2 ,1 ,5]
    simulate_full_solve_beam(model, sample_state, max_steps=100, beam_width=3, alpha=0.6)