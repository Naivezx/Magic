import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio
import numpy as np
from Magic_Test_Resnet import apply_move

# Face codes
L, U, F, D, R, B = 0, 9, 18, 27, 36, 45

# Color mapping
colors = ['white', 'red', 'blue', 'yellow', 'green', 'orange']

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

# Interactive cube display function
def show_interactive_cube(state, title=''):
    """Display a cube that can be rotated with mouse"""
    fig, ax = draw_cube(state, title=title, interactive=True)
    
    # Enable interactive mode
    plt.ion()
    
    # Display instructions
    print("🎮 Interactive Rubik's Cube Viewer")
    print("• Drag mouse to rotate cube")
    print("• Mouse wheel to zoom")
    print("• Close window to exit")
    
    plt.show()
    plt.ioff()
    return fig, ax

# ====== Initial state and moves ======
initial_state = list(map(int, '2, 0, 5, 2, 0, 5, 3, 5, 1, 0, 3, 0, 3, 1, 5, 0, 3, 4, 3, 5, 5, 1, 2, 2, 4, 4, 0, 2, 1, 2, 4, 3, 4, 4, 0, 4, 1, 0, 1, 3, 4, 1, 1, 2, 3, 5, 4, 3, 2, 5, 0, 2, 1, 5'.split(',')))
moves = "F D B2 D F D B' U B2 D U B' U' B R U' R' U F' U' F U L U2 L' U B' U' B U2 F U2 F' U2 F U' F' U F' L F U F' U' L' U2 F' U' R' F' R2 U' R' U R' F R F".split()

# ====== Simulate all states ======
os.makedirs("photo", exist_ok=True)
images = []
states = [initial_state]
curr_state = initial_state.copy()

for move in moves:
    curr_state = apply_move(curr_state, move)
    states.append(curr_state)

print("✅ Images and GIF generation completed!")

# ====== Save images and create animation ======
for i, state in enumerate(states):
    filename = f"photo/step_{i:03d}.png"
    move = "start" if i == 0 else moves[i-1]
    draw_cube(state, title=f"Step {i}: {move}", save_path=filename)
    images.append(filename)

with imageio.get_writer("photo/solution.gif", mode='I', duration=1.2) as writer:
    for filename in images:
        writer.append_data(imageio.imread(filename))

# ====== Interactive viewer ======
current_step = 0

def interactive_viewer():
    """Interactive Rubik's cube viewer"""
    global current_step
    
    print("\n🎮 Starting interactive Rubik's cube viewer...")
    print("📍 Instructions:")
    print("  • Drag mouse to rotate cube")
    print("  • Press SPACE to go to next step")
    print("  • Press 'b' to go to previous step")
    print("  • Press 'r' to reset to initial state")
    print("  • Close window to exit")
    
    fig, ax = draw_cube(states[current_step], 
                       title=f"Step {current_step}: {'Initial' if current_step == 0 else moves[current_step-1]}", 
                       interactive=True)
    
    def on_key(event):
        global current_step
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
        
        # Draw all six faces - NO GAPS
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
        
        move_name = "Initial" if current_step == 0 else moves[current_step-1]
        plt.title(f"Step {current_step}: {move_name}", fontsize=14, pad=20)
        fig.canvas.draw()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

# Start interactive viewer
interactive_viewer()