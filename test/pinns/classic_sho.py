"""
finding the solution to the classical harmonic oscillator:

```
y''(t) + y = 0
y(0) = 1
y'(0) = 0
```

using a PINN.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

class MLP(torch.nn.Module):
    def __init__(self, n_in, n_out, hidden):
        super(MLP, self).__init__()
        N_hidden = len(hidden)
        self.MLP_layer = torch.nn.Sequential(
            torch.nn.Linear(n_in, hidden[0]), torch.nn.Tanh(),
            *[torch.nn.Sequential(torch.nn.Linear(hidden[i], hidden[i+1]), torch.nn.Tanh()) for i in range(1, N_hidden - 1)],
            torch.nn.Linear(hidden[-1], n_out)
        )

    def forward(self, t):
        after_MLP = self.MLP_layer(t)
        return after_MLP
#-------
    

Npts = 1000
t_bulk = torch.linspace(0, 5, Npts, requires_grad=True).unsqueeze(dim=1)

n_in = 1
n_out = 1
hidden = [32, 32, 32]
model = MLP(n_in, n_out, hidden)

N_epochs = 1000

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
t0 = torch.tensor([[0.0]], requires_grad=True)

t_np = t_bulk.detach().numpy()
y_exact = np.cos(t_np)
plt.plot(t_np, y_exact, label="exact")

model.train() # training mode
for i in range(N_epochs):
    optimizer.zero_grad()
    y = model(t_bulk)
    dy_dt = torch.autograd.grad(y, t_bulk, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    d2y_dt2 = torch.autograd.grad(dy_dt, t_bulk,  grad_outputs=torch.ones_like(dy_dt), create_graph=True)[0]
    LHS = d2y_dt2 + y
    RHS = 0
    loss_bulk = ((LHS - RHS)**2).mean()
    y0 = model(t0)
    dy_dt0 = torch.autograd.grad(y0, t0, grad_outputs=torch.ones_like(y0), create_graph=True)[0] 
    # initial condition y(0)=1
    loss_boundary = ((y0 - 1.0)**2).mean() + ((dy_dt0 - 0.0)**2).mean()
    loss = loss_bulk + loss_boundary
    loss.backward()
    optimizer.step()

    if (100*i/N_epochs) % 10 == 0:
        loss_MSE = ((y-torch.tensor(y_exact))**2).mean()
        print(f"Epoch: {i}/{N_epochs} | loss={loss.item()} loss_bulk={loss_bulk.item()}, loss_t0={loss_boundary.item()} | MSE={loss_MSE}") # , loss_bulk={loss_bulk.item()}, loss_boundary={loss_boundary.item()}")
        # plotting the intermediate results
        model.eval()
        y_prediction = model(t_bulk).detach().numpy()
        plt.plot(t_np, y_prediction, label=f"i={i}", color="blue", alpha=i/N_epochs)
        model.train()
#    

plt.legend()
plt.show()
    



    
