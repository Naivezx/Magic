import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse
from heapq import heappush
from collections import Counter

# Add visualization imports
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio

SAMPLE_STATE = [4, 0, 2, 3, 0, 2, 1, 5, 0, 3, 1, 2, 1, 1, 2, 1, 5, 2, 4, 0, 0, 4, 2, 1, 1, 0, 5, 2, 2, 1, 1, 3, 5, 0, 3, 5, 3, 3, 3, 4, 4, 3, 4, 4, 0, 4, 2, 5, 5, 5, 4, 3, 0, 5]
MAX_STEPS = 100
BEAM_WIDTH = 3
DIFFUSION_DISTANCE = 0.5

# ----------------------
# Constants
# ----------------------
FACE_SIZE = 54
NUM_CLASSES = 18
MOVE_LABELS = ['U', 'U\'', 'U2', 'D', 'D\'', 'D2', 'L', 'L\'', 'L2', 'R', 'R\'', 'R2', 'F', 'F\'', 'F2', 'B', 'B\'', 'B2']
MOVE_TO_IDX = {m: i for i, m in enumerate(MOVE_LABELS)}
IDX_TO_MOVE = {i: m for m, i in MOVE_TO_IDX.items()}
L, U, F, D, R, B = range(0, 54, 9)  

# Color mapping for visualization
colors = ['white', 'red', 'blue', 'yellow', 'green', 'orange']

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
        out += self.shortcut(identity)
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
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(64, 64, stride=1),
            ResidualBlock(64, 128, stride=1),
            ResidualBlock(128, 256, stride=1)
        )
        
        self.flatten = nn.Flatten()
        
        self.mlp = nn.Sequential(
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )
    
    def forward(self, x):
        original_training = self.training
        if x.size(0) == 1:
            self.eval()
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = False
        
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.flatten(x)
        x = self.mlp(x)
        
        if x.size(0) == 1:
            self.train(original_training)
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = True
        
        return x

def evaluate_model(model):
    device = next(model.parameters()).device
    dataset = CubeDataset('test')
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

def diffusion_distance(state):
    solved_state = [i//9 for i in range(54)]
    mismatches = sum(1 for s, t in zip(state, solved_state) if s != t)
    cross_penalty = sum(1 for i in [12, 13, 14] if state[i] != state[13]) / 4.0  # U face: 9-17
    corner_penalty = sum(1 for i in [9, 11, 15, 17] if state[i] != state[13]) / 4.0
    return 0.5 * (mismatches / 54.0) + 0.3 * cross_penalty + 0.2 * corner_penalty

def predict_move(model, state):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = torch.tensor(state, dtype=torch.float32).view(1, 6, 3, 3).to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
        return IDX_TO_MOVE[pred]

def rotate_face(face, direction='clockwise'):
    if direction == 'clockwise':
        return [face[6], face[3], face[0], face[7], face[4], face[1], face[8], face[5], face[2]]
    else:
        return [face[2], face[5], face[8], face[1], face[4], face[7], face[0], face[3], face[6]]

def apply_move(state, move):
    new_state = state.copy()
    
    if move == 'U':
        new_state[U:U+9] = rotate_face(state[U:U+9], 'clockwise')
        new_state[F:F+3], new_state[R:R+3], new_state[B:B+3], new_state[L:L+3] = \
            state[R:R+3], state[B:B+3], state[L:L+3], state[F:F+3]
    elif move == 'U\'':
        new_state[U:U+9] = rotate_face(state[U:U+9], 'counterclockwise')
        new_state[F:F+3], new_state[R:R+3], new_state[B:B+3], new_state[L:L+3] = \
            state[L:L+3], state[F:F+3], state[R:R+3], state[B:B+3]
    elif move == 'U2':
        new_state[U:U+9] = rotate_face(rotate_face(state[U:U+9], 'clockwise'), 'clockwise')
        new_state[F:F+3], new_state[R:R+3], new_state[B:B+3], new_state[L:L+3] = \
            state[B:B+3], state[L:L+3], state[F:F+3], state[R:R+3]
    elif move == 'D':
        new_state[D:D+9] = rotate_face(state[D:D+9], 'clockwise')
        temp = state[F+6:F+9]
        new_state[F+6:F+9] = state[L+6:L+9]
        new_state[L+6:L+9] = state[B+6:B+9]
        new_state[B+6:B+9] = state[R+6:R+9]
        new_state[R+6:R+9] = temp
    elif move == 'D\'':
        new_state[D:D+9] = rotate_face(state[D:D+9], 'counterclockwise')
        temp = state[F+6:F+9]
        new_state[F+6:F+9] = state[R+6:R+9]
        new_state[R+6:R+9] = state[B+6:B+9]
        new_state[B+6:B+9] = state[L+6:L+9]
        new_state[L+6:L+9] = temp
    elif move == 'D2':
        new_state[D:D+9] = rotate_face(rotate_face(state[D:D+9], 'clockwise'), 'clockwise')
        temp1 = state[F+6:F+9]
        temp2 = state[R+6:R+9]
        new_state[F+6:F+9] = state[B+6:B+9]
        new_state[R+6:R+9] = state[L+6:L+9]
        new_state[B+6:B+9] = temp1
        new_state[L+6:L+9] = temp2
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

def simplify_move_sequence(moves):
    if not moves:
        return []
    
    simplified = []
    i = 0
    faces = ['U', 'D', 'L', 'R', 'F', 'B']
    
    while i < len(moves):
        if i + 1 < len(moves):
            curr_move = moves[i]
            next_move = moves[i + 1]
            curr_face = curr_move.rstrip("'2")
            next_face = next_move.rstrip("'2")
            
            if curr_face == next_face and (
                (curr_move == curr_face and next_move == curr_face + "'") or
                (curr_move == curr_face + "'" and next_move == curr_face)
            ):
                i += 2
                continue
            
            if curr_face == next_face and curr_move == next_move and curr_move in [curr_face, curr_face + "'"]:
                simplified.append(curr_face + "2")
                i += 2
                continue
            
            if curr_face == next_face and curr_move == curr_face + "2" and next_move == curr_face + "2":
                i += 2
                continue
        
        if i + 2 < len(moves):
            curr_move = moves[i]
            next_move = moves[i + 1]
            next_next_move = moves[i + 2]
            curr_face = curr_move.rstrip("'2")
            if curr_move == next_move == next_next_move and curr_move in [curr_face, curr_face + "'"]:
                simplified.append(curr_face + "'" if curr_move == curr_face else curr_face)
                i += 3
                continue
        
        if i + 1 < len(moves):
            curr_move = moves[i]
            next_move = moves[i + 1]
            curr_face = curr_move.rstrip("'2")
            next_face = next_move.rstrip("'2")
            if curr_face == next_face and (
                (curr_move == curr_face + "2" and next_move == curr_face) or
                (curr_move == curr_face and next_move == curr_face + "2")
            ):
                simplified.append(curr_face + "'")
                i += 2
                continue
            if curr_face == next_face and (
                (curr_move == curr_face + "2" and next_move == curr_face + "'") or
                (curr_move == curr_face + "'" and next_move == curr_face + "2")
            ):
                simplified.append(curr_face)
                i += 2
                continue
        
        if i + 2 < len(moves):
            curr_move = moves[i]
            next_move = moves[i + 1]
            next_next_move = moves[i + 2]
            curr_face = curr_move.rstrip("'2")
            if curr_move == curr_face and next_move == curr_face + "'" and next_next_move == curr_face:
                simplified.append(curr_face)
                i += 3
                continue
        
        simplified.append(moves[i])
        i += 1
    
    if simplified != moves:
        return simplify_move_sequence(simplified)
    return simplified

def simulate_full_solve_beam(model, state, max_steps=2, beam_width=10, alpha=0.6):
    print("Start solving...")
    device = next(model.parameters()).device
    initial_dist = diffusion_distance(state)
    beams = [(0.0, state.copy(), [])]
    
    scores = []
    distances = []
    seen_states = set()
    
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
                
                new_history = simplify_move_sequence(move_history + [action])
                if not new_history:
                    continue
                if new_history == move_history:
                    new_history = move_history + [action]
                
                new_state = current_state
                for move in new_history[len(move_history):]:
                    new_state = apply_move(new_state, move)
                
                state_hash = tuple(new_state)
                if state_hash in seen_states:
                    continue
                seen_states.add(state_hash)
                
                new_dist = diffusion_distance(new_state)
                
                dist_score = new_dist / (initial_dist + 1e-6)
                prob_score = -np.log(p.item() + 1e-6)
                new_score = alpha * prob_score + (1 - alpha) * dist_score
                
                if is_solved(new_state):
                    print("Solved!")
                    print(f"Solution sequence: {' '.join(new_history)}")
                    print(f"Steps: {len(new_history)}")
                    scores.append(new_score)
                    distances.append(new_dist)
                    return new_history, scores, distances
                
                if len(new_history) >= 4:
                    if new_history[-4:] == [new_history[-1]] * 4:
                        print(f"Skipping repetitive move: {new_history[-4:]}")
                        continue
                    if new_history[-4:] in [
                        [new_history[-4], new_history[-3], new_history[-4], new_history[-3]],
                        [new_history[-3], new_history[-4], new_history[-3], new_history[-4]]
                    ] and new_history[-4] in [new_history[-3].replace("'", "") + "'", new_history[-3].replace("'", "")] + [new_history[-3] + "2"]:
                        print(f"Skipping alternating pattern: {new_history[-4:]}")
                        continue
                
                heappush(new_beams, (new_score, new_state, new_history))
        
        beams = new_beams[:beam_width]
        if beams:
            scores.append(beams[0][0])
            distances.append(diffusion_distance(beams[0][1]))
            print(f"Step {step + 1}: Best sequence: {' '.join(beams[0][2])}, Distance: {distances[-1]:.4f}, Score: {scores[-1]:.4f}")
        else:
            print("All beams filtered due to cycles, stopping.")
            break
        
        beams = [b for b in beams if not (
            len(b[2]) >= 4 and (
                b[2][-4:] == [b[2][-1]] * 4 or
                (b[2][-4:] in [
                    [b[2][-4], b[2][-3], b[2][-4], b[2][-3]],
                    [b[2][-3], b[2][-4], b[2][-3], b[2][-4]]
                ] and b[2][-4] in [b[2][-3].replace("'", "") + "'", b[2][-3].replace("'", "")] + [b[2][-3] + "2"])
            )
        )]
        if not beams:
            print("All beams in cycle, stopping.")
            break
    
    print("Max steps reached. Could not solve fully.")
    if beams:
        _, _, best_history = beams[0]
        print(f"Best attempted sequence: {' '.join(best_history)}")
        print(f"Final distance: {diffusion_distance(beams[0][1]):.4f}")
        return best_history, scores, distances
    else:
        print("No valid sequences found.")
        return [], scores, distances

# ====== Visualization Functions ======
def draw_cube(state, title='', save_path=None, interactive=False):
    assert len(state) == 54, f"State length error: {len(state)}"
    assert all(0 <= x < len(colors) for x in state), f"Invalid color code in state: {state}"

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=45)
    ax.set_axis_off()

    def draw_face_grid(face_start, face_normal, face_right, face_up, face_center):
        """Draw a 3x3 grid for one face without gaps"""
        for i in range(3):
            for j in range(3):
                idx = face_start + i * 3 + j
                color = colors[state[idx]] if 0 <= state[idx] < len(colors) else 'gray'
                
                # Calculate position for each small cube - NO GAPS
                offset_x = (j - 1) * 0.333
                offset_y = (1 - i) * 0.333
                
                # Create actual 3D cube for each cell
                cube_center = face_center + offset_x * face_right + offset_y * face_up
                
                # Create a 3D cube with proper depth - LARGER SIZE FOR NO GAPS
                size = 0.167  # Increased size to eliminate gaps
                
                # Front face of the small cube - directly on the face
                front_corners = [
                    cube_center + size * (-face_right - face_up),
                    cube_center + size * (face_right - face_up),
                    cube_center + size * (face_right + face_up),
                    cube_center + size * (-face_right + face_up)
                ]
                
                # Add the front face (main colored face)
                poly_front = Poly3DCollection([front_corners], facecolors=color, 
                                            edgecolors='black', linewidths=0.5, alpha=1.0)
                ax.add_collection3d(poly_front)

    # Define the six faces of the cube - NO GAPS between faces
    # Front face (F) - exactly at z = 0.5
    draw_face_grid(F, np.array([0, 0, 1]), np.array([1, 0, 0]), 
                   np.array([0, 1, 0]), np.array([0, 0, 0.5]))
    
    # Right face (R) - exactly at x = 0.5
    draw_face_grid(R, np.array([1, 0, 0]), np.array([0, 0, -1]), 
                   np.array([0, 1, 0]), np.array([0.5, 0, 0]))
    
    # Top face (U) - exactly at y = 0.5
    draw_face_grid(U, np.array([0, 1, 0]), np.array([1, 0, 0]), 
                   np.array([0, 0, -1]), np.array([0, 0.5, 0]))
    
    # Back face (B) - exactly at z = -0.5
    draw_face_grid(B, np.array([0, 0, -1]), np.array([-1, 0, 0]), 
                   np.array([0, 1, 0]), np.array([0, 0, -0.5]))
    
    # Left face (L) - exactly at x = -0.5
    draw_face_grid(L, np.array([-1, 0, 0]), np.array([0, 0, 1]), 
                   np.array([0, 1, 0]), np.array([-0.5, 0, 0]))
    
    # Bottom face (D) - exactly at y = -0.5
    draw_face_grid(D, np.array([0, -1, 0]), np.array([1, 0, 0]), 
                   np.array([0, 0, 1]), np.array([0, -0.5, 0]))

    # Set axis ranges
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])

    if title:
        plt.title(title, fontsize=14, pad=20)
        
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close()
    elif not interactive:
        plt.show()
        
    return fig, ax

def create_visualization(initial_state, solution_moves):
    """Create visualization with images and GIF for the solution"""
    print("\nCreating visualization...")
    
    # Create photo directory
    os.makedirs("photo", exist_ok=True)
    images = []
    states = [initial_state]
    curr_state = initial_state.copy()

    # Generate all states
    for move in solution_moves:
        curr_state = apply_move(curr_state, move)
        states.append(curr_state.copy())

    print(f"Generating {len(states)} images...")

    # Save images for each step
    for i, state in enumerate(states):
        filename = f"photo/step_{i:03d}.png"
        move = "Initial State" if i == 0 else solution_moves[i-1]
        draw_cube(state, title=f"Step {i}: {move}", save_path=filename)
        images.append(filename)

    # Create GIF animation
    print("Creating GIF animation...")
    with imageio.get_writer("photo/solution.gif", mode='I', duration=1.2) as writer:
        for filename in images:
            writer.append_data(imageio.imread(filename))

def interactive_viewer(states, moves):
    """Interactive Rubik's cube viewer"""
    current_step = 0
    
    print("\n🎮 Starting interactive Rubik's cube viewer...")
    print("📍 Instructions:")
    print("  • Drag mouse to rotate cube")
    print("  • Press SPACE to go to next step")
    print("  • Press 'b' to go to previous step")
    print("  • Press 'r' to reset to initial state")
    print("  • Close window to exit")
    
    fig, ax = draw_cube(states[current_step], 
                       title=f"Step {current_step}: {'Initial State' if current_step == 0 else moves[current_step-1]}", 
                       interactive=True)
    
    def on_key(event):
        nonlocal current_step
        if event.key == ' ':  # Space key: next step
            current_step = (current_step + 1) % len(states)
            update_display()
        elif event.key == 'b':  # b key: previous step
            current_step = (current_step - 1) % len(states)
            update_display()
        elif event.key == 'r':  # r key: reset
            current_step = 0
            update_display()
    
    def update_display():
        ax.clear()
        ax.set_axis_off()
        
        # Redraw cube
        state = states[current_step]
        
        def draw_face_grid(face_start, face_normal, face_right, face_up, face_center):
            for i in range(3):
                for j in range(3):
                    idx = face_start + i * 3 + j
                    color = colors[state[idx]] if 0 <= state[idx] < len(colors) else 'gray'
                    
                    offset_x = (j - 1) * 0.333
                    offset_y = (1 - i) * 0.333
                    cube_center = face_center + offset_x * face_right + offset_y * face_up
                    
                    size = 0.167
                    
                    front_corners = [
                        cube_center + size * (-face_right - face_up),
                        cube_center + size * (face_right - face_up),
                        cube_center + size * (face_right + face_up),
                        cube_center + size * (-face_right + face_up)
                    ]
                    
                    poly_front = Poly3DCollection([front_corners], facecolors=color, 
                                                edgecolors='black', linewidths=0.5, alpha=1.0)
                    ax.add_collection3d(poly_front)
        
        # Draw all six faces
        draw_face_grid(F, np.array([0, 0, 1]), np.array([1, 0, 0]), 
                       np.array([0, 1, 0]), np.array([0, 0, 0.5]))
        draw_face_grid(R, np.array([1, 0, 0]), np.array([0, 0, -1]), 
                       np.array([0, 1, 0]), np.array([0.5, 0, 0]))
        draw_face_grid(U, np.array([0, 1, 0]), np.array([1, 0, 0]), 
                       np.array([0, 0, -1]), np.array([0, 0.5, 0]))
        draw_face_grid(B, np.array([0, 0, -1]), np.array([-1, 0, 0]), 
                       np.array([0, 1, 0]), np.array([0, 0, -0.5]))
        draw_face_grid(L, np.array([-1, 0, 0]), np.array([0, 0, 1]), 
                       np.array([0, 1, 0]), np.array([-0.5, 0, 0]))
        draw_face_grid(D, np.array([0, -1, 0]), np.array([1, 0, 0]), 
                       np.array([0, 0, 1]), np.array([0, -0.5, 0]))
        
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_box_aspect([1,1,1])
        
        move_name = "Initial State" if current_step == 0 else moves[current_step-1]
        plt.title(f"Step {current_step}: {move_name}", fontsize=14, pad=20)
        fig.canvas.draw()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

# ----------------------
# Main
# ----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Rubik's Cube Solver")
    parser.add_argument('--model_path', type=str, default='cube_model.pt',
                        help="Path to the trained model file")
    parser.add_argument('--visualize', action='store_true', 
                        help="Create visualization after solving")
    parser.add_argument('--interactive', action='store_true',
                        help="Show interactive viewer after solving")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CubeSolverCNN().to(device)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    except RuntimeError as e:
        print(f"Error loading model from {args.model_path}: {e}")
        exit(1)
    
    model.eval()
    evaluate_model(model)

    # Sample state from Magic_Test_Resnet.py
    sample_state = SAMPLE_STATE
    
    # Solve the cube
    result, scores, distances = simulate_full_solve_beam(model, sample_state, MAX_STEPS, BEAM_WIDTH, DIFFUSION_DISTANCE)
    
    if not scores or not distances:
        print("Warning: No data to plot. Scores and distances lists are empty.")
    else:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(scores, label='Score')
        plt.xlabel('Step')
        plt.ylabel('Score')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(distances, label='Distance')
        plt.xlabel('Step')
        plt.ylabel('Distance')
        plt.legend()
        plt.savefig('search_metrics.png')
        plt.close()
        print("Generated search_metrics.png with Score and Distance plots.")

    # If solved successfully, create visualization
    if result and (args.visualize or args.interactive):
        print(f"\n🎯 Solution found with {len(result)} moves!")
        print(f"Moves: {' '.join(result)}")
        
        # Generate all states for visualization
        states = [sample_state]
        curr_state = sample_state.copy()
        for move in result:
            curr_state = apply_move(curr_state, move)
            states.append(curr_state.copy())
        
        if args.visualize:
            create_visualization(sample_state, result)
        
        if args.interactive:
            interactive_viewer(states, result)