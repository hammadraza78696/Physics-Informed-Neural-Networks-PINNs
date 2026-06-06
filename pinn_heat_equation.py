import torch
import torch.nn as nn
import numpy as np

# 1. Define the Neural Network Architecture
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # Input: space (x) and time (t) -> Output: temperature u(x,t)
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# 2. Physics-Informed Loss Calculation
def physics_loss(model, x_domain, t_domain, alpha=0.01):
    x_domain.requires_grad_(True)
    t_domain.requires_grad_(True)
    
    u = model(x_domain, t_domain)
    
    # Compute first-order derivative with respect to time (du/dt)
    u_t = torch.autograd.grad(u, t_domain, torch.ones_like(u), create_graph=True)
    
    # Compute first and second-order derivatives with respect to space (du/dx, d^2u/dx^2)
    u_x = torch.autograd.grad(u, x_domain, torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, x_domain, torch.ones_like(u_x), create_graph=True)
    
    # Heat Equation Residual: f = du/dt - alpha * d^2u/dx^2
    residual = u_t - alpha * u_xx
    return torch.mean(residual**2)

# 3. Execution Driver
if __name__ == "__main__":
    print("Initializing PINN Optimization Network...")
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Generate random interior domain points for collocation boundary verification
    x_collocation = torch.rand(100, 1)
    t_collocation = torch.rand(100, 1)
    
    # Quick trial training iteration loop
    for epoch in range(11):
        optimizer.zero_grad()
        loss = physics_loss(model, x_collocation, t_collocation)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d} | Mean Squared Physics Residual Loss: {loss.item():.6f}")
    
    print("PINN Framework compilation successful!")
