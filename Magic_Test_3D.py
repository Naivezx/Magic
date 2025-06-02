import os
os.environ["KERAS_BACKEND"] = "torch"  # Ensure PyTorch backend
import torch
import numpy as np
import keras_core as keras
import keras_core.layers as layers
import keras_core.models as models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import argparse
import random

# Constants
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

def evaluate_model(model, test_folder='test'):
    dataset = CubeDataset(test_folder)
    loader = DataLoader(dataset, batch_size=128)
    correct, total = 0, 0
    model.evaluate()  # Ensure evaluation mode
    for x, y in tqdm(loader, desc="Evaluating"):
        x_np, y_np = x.numpy(), y.numpy()
        predictions = model.predict(x_np, batch_size=x_np.shape[0], verbose=0)
        preds = np.argmax(predictions, axis=1)
        correct += (preds == y_np).sum()
        total += y_np.shape[0]
    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

def diffusion_distance(state):
    solved_state = [i//9 for i in range(54)]
    mismatches = sum(1 for s, t in zip(state, solved_state) if s != t)
    cross_penalty = sum(1 for i in [3, 4, 5] if state[i] != state[4]) / 4.0
    corner_penalty = sum(1 for i in [0, 2, 6, 8] if state[i] != state[4]) / 4.0
    return 0.5 * (mismatches / 54.0) + 0.3 * cross_penalty + 0.2 * corner_penalty

def predict_move(model, state):
    model.evaluate()  # Ensure evaluation mode
    x = np.array(state, dtype=np.float32).reshape(1, 1, 6, 3, 3)
    predictions = model.predict(x, verbose=0)
    pred = np.argmax(predictions, axis=1)[0]
    return IDX_TO_MOVE[pred]

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

def simulate_full_solve_beam(model, state, max_steps=200, beam_width=10, alpha=0.6):
    print("Start solving...")
    initial_dist = diffusion_distance(state)
    beams = [(0.0, state.copy(), [])]
    
    scores = []
    distances = []
    
    for step in range(max_steps):
        new_beams = []
        for score, current_state, move_history in beams:
            x = np.array(current_state, dtype=np.float32).reshape(1, 1, 6, 3, 3)
            predictions = model.predict(x, verbose=0)
            probs = predictions[0]
            top_indices = np.argsort(probs)[-beam_width:]
            top_probs = probs[top_indices]
            
            for p, a in zip(top_probs, top_indices):
                if random.random() < 0.1:  # 10% probability for random action
                    a = random.choice(list(range(NUM_CLASSES)))
                else:
                    a = a
                action = IDX_TO_MOVE[a]
                if len(move_history) > 0:
                    last_move = move_history[-1]
                    opposite_moves = {
                        'U': ['U\'', 'U2'], 'U\'': ['U', 'U2'], 'U2': ['U', 'U\''],
                        'D': ['D\'', 'D2'], 'D\'': ['D', 'D2'], 'D2': ['D', 'D\''],
                        'L': ['L\'', 'L2'], 'L\'': ['L', 'L2'], 'L2': ['L', 'L\''],
                        'R': ['R\'', 'R2'], 'R\'': ['R', 'R2'], 'R2': ['R', 'R\''],
                        'F': ['F\'', 'F2'], 'F\'': ['F', 'F2'], 'F2': ['F', 'F\''],
                        'B': ['B\'', 'B2'], 'B\'': ['B', 'B2'], 'B2': ['B', 'B\'']
                    }
                    # Skip opposite moves (commented out as in original)
                    # if action in opposite_moves.get(last_move, []):
                    #     continue
                new_state = apply_move(current_state, action)
                new_dist = diffusion_distance(new_state)
                
                dist_score = new_dist / (initial_dist + 1e-6)
                prob_score = -np.log(p + 1e-6)
                new_score = alpha * prob_score + (1 - alpha) * dist_score
                
                if is_solved(new_state):
                    print("Solved!")
                    print(f"Solution sequence: {' '.join(move_history + [action])}")
                    print(f"Steps: {len(move_history) + 1}")
                    scores.append(new_score)
                    distances.append(new_dist)
                    return move_history + [action], scores, distances
                
                new_history = move_history + [action]
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
                
                new_beams.append((new_score, new_state, new_history))
        
        new_beams.sort(key=lambda x: x[0])
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Rubik's Cube Solver with 3D CNN")
    parser.add_argument('--model_path', type=str, default='cube_model_epoch_20.weights.h5',
                        help="Path to the trained model weights file")
    args = parser.parse_args()

    # Create and load model
    model = create_cube_solver_model()
    try:
        model.load_weights(args.model_path)
        print(f"Loaded model weights from {args.model_path}")
    except Exception as e:
        print(f"Error loading model weights from {args.model_path}: {e}")
        exit(1)

    # # Evaluate model accuracy
    # accuracy = evaluate_model(model, test_folder='test')
    # print(f"Model accuracy: {accuracy:.2f}%")

    # Test with a sample state
    sample_state = [2,0,5,2,0,5,3,5,1,0,3,0,3,1,5,0,3,4,3,5,5,1,2,2,4,4,0,2,1,2,4,3,4,4,0,4,1,0,1,3,4,1,1,2,3,5,4,3,2,5,0,2,1,5]
    print("Verifying sample state:", Counter(sample_state))

    # Run beam search with visualization
    result, scores, distances = simulate_full_solve_beam(model, sample_state, max_steps=200, beam_width=18, alpha=1)

    # Plot results
    if not scores or not distances:
        print("Warning: No data to plot. Scores and distances lists are empty.")
    else:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(scores, label='Score')
        plt.xlabel('Step')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(distances, label='Distance')
        plt.xlabel('Step')
        plt.ylabel('Distance')
        plt.legend()
        plt.grid(True)
        plt.savefig('search_metrics.png')
        plt.close()
        print("Generated search_metrics.png with Score and Distance plots.")