import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Custom Sine Activation Layer
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega_0=1.0): # Set to 1.0
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Proper initialization is critical
        nn.init.uniform_(self.linear.weight, -1.0 / in_features, 1.0 / in_features)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

# 2. Unified Network
class OscillatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Using a slightly wider network helps with convergence
        self.net = nn.Sequential(
            SineLayer(1, 64, omega_0=1.0),
            SineLayer(64, 64, omega_0=1.0),
            nn.Linear(64, 2) 
        )
    def forward(self, t):
        return self.net(t)

# Setup
Npts = 500 # Increased sampling
t_bulk = torch.linspace(0, 2*np.pi, Npts, requires_grad=True).reshape(-1, 1)
y_true = torch.cos(t_bulk)

model = OscillatorNet()
# Adam is great, but sometimes a slightly smaller LR helps at the end
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# 3. Training Loop
N_epochs = 1500 # Added more epochs to allow the lower frequency to settle
print("Starting Training...")

for i in range(N_epochs + 1):
    optimizer.zero_grad()
    
    outputs = model(t_bulk)
    y = outputs[:, 0:1]
    v = outputs[:, 1:2]
    
    dy_dt = torch.autograd.grad(y, t_bulk, torch.ones_like(y), create_graph=True)[0]
    dv_dt = torch.autograd.grad(v, t_bulk, torch.ones_like(v), create_graph=True)[0]
    
    # Residuals
    loss_physics = ((dv_dt + y)**2).mean() + ((v - dy_dt)**2).mean()
    
    # Boundary Conditions
    t0 = torch.zeros(1, 1)
    y0v0 = model(t0)
    loss_bc = 100 * ((y0v0[0, 0] - 1.0)**2 + (y0v0[0, 1] - 0.0)**2)
    
    loss = loss_physics + loss_bc
    loss.backward()
    optimizer.step()
    
    if 100 * i / N_epochs % 10 == 0:
        mse = torch.mean((y - y_true)**2).item()
        print(f"Epoch {i:4d} | Physics: {loss_physics.item():.6f} | MSE: {mse:.6f}")

# Plotting
y_pred = model(t_bulk)[:, 0].detach().numpy()
plt.figure(figsize=(10, 5))
plt.plot(t_bulk.detach(), y_true.detach(), 'k--', label="Exact")
plt.plot(t_bulk.detach(), y_pred, label="PINN (Fixed omega_0)")

plt.scatter(t_bulk.detach(), np.zeros_like(t_bulk.detach().numpy()), marker=".")
plt.legend()
plt.show()
