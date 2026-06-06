import torch
import numpy as np

def generate_collocation_mesh(x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0, num_points=200):
    """
    Generates structured spatial and temporal boundary coordinates 
    for PDE numeric evaluation mesh layers.
    """
    x_raw = np.linspace(x_min, x_max, num_points)
    t_raw = np.linspace(t_min, t_max, num_points)
    
    X_mesh, T_mesh = np.meshgrid(x_raw, t_raw)
    
    x_tensor = torch.tensor(X_mesh.flatten(), dtype=torch.float32).unsqueeze(1)
    t_tensor = torch.tensor(T_mesh.flatten(), dtype=torch.float32).unsqueeze(1)
    
    return x_tensor, t_tensor
