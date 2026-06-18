import torch
import torch.nn as nn
import torch.optim as optim

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Hardware Routing
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Initializing on Device: {device}")

# ==========================================
# 2. The Architecture (Single-Threat Oracle)
# ==========================================
class NeuralCBF(nn.Module):
    def __init__(self):
        super(NeuralCBF, self).__init__()
        # Input: [dx, dy], Output: h(x)
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. Fast Batched Data Generation
# ==========================================
def generate_relative_data(num_samples=100000):
    print(f"[INFO] Generating {num_samples} relative data points...")
    # Train the network to understand relative threats up to 10 meters away
    dX = np.random.uniform(-10.0, 10.0, (num_samples, 2))
    
    # Fixed obstacle radius
    obstacle_radius = 1.5
    
    # Calculate analytical safety bounds: positive = safe, negative = unsafe
    distances = np.linalg.norm(dX, axis=1)
    h_values = distances - obstacle_radius
    
    # Push data directly to VRAM
    inputs = torch.tensor(dX, dtype=torch.float32).to(device)
    targets = torch.tensor(h_values, dtype=torch.float32).unsqueeze(1).to(device)
    
    return inputs, targets

# ==========================================
# 4. Main Training & Visualization Execution
# ==========================================
if __name__ == "__main__":
    
    # --- PHASE A: TRAINING ---
    X_train, y_train = generate_relative_data(100000)
    
    model = NeuralCBF().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    print("[INFO] Executing Full-Batch GPU Training...")
    epochs = 1500
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # PyTorch processes all 100,000 points simultaneously here
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 250 == 0:
            print(f"Epoch {epoch}/{epochs} - MSE Loss: {loss.item():.5f}")
            
    # Move weights back to system memory before saving
    torch.save(model.cpu().state_dict(), "ncbf_batched_weights.pth")
    print("[SUCCESS] Multi-Agent Ready Model saved to 'ncbf_batched_weights.pth'")
    
    # --- PHASE B: VISUALIZATION ---
    print("[INFO] Rendering the learned ego-centric safety boundary...")
    model.eval() 
    model.to(device) # Push back to GPU for the grid inference
    
    # Create a grid from -10 to 10 meters representing the robot's relative vision
    x_grid, y_grid = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    grid_points = np.c_[x_grid.ravel(), y_grid.ravel()]
    
    # Push grid to GPU
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # Infer on GPU, pull results back to CPU for Matplotlib
        h_pred = model(grid_tensor).cpu().numpy().reshape(100, 100)
        
    plt.figure(figsize=(8, 6))
    
    # Draw the contour gradient
    cp = plt.contourf(x_grid, y_grid, h_pred, levels=50, cmap='RdYlGn')
    plt.colorbar(cp, label="Learned h(x) [Positive = Safe, Negative = Crash]")
    
    # Draw the strict mathematical boundary (h = 0)
    plt.contour(x_grid, y_grid, h_pred, levels=[0.0], colors='black', linewidths=3)
    
    # The obstacle is always at the center of the ego-centric view
    plt.scatter(0.0, 0.0, color='red', label="Relative Obstacle Center", marker='x', s=100)
    
    plt.title("GPU-Trained Batched Neural CBF: 10m Vision Radius")
    plt.xlabel("dx (Relative X Distance from Robot)")
    plt.ylabel("dy (Relative Y Distance from Robot)")
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()