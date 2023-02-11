import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Define the function to generate the data
def generate_data(x):
    return np.sin(x) + np.cos(2 * x)

# Define the neural network
class LinearNetwork(nn.Module):
    def __init__(self):
        super(LinearNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the loss function
def loss_fn(u_pred, u_target):
    loss = ((u_pred - u_target)**2).mean()
    return loss

# Define the initial conditions
x = np.linspace(0, 2*np.pi, 100)
u0 = generate_data(x)
u1 = np.cos(x) - np.sin(2 * x)

# Define the network and the optimizer
model = LinearNetwork()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Define the number of steps and the time step
steps = 1000
dt = 0.01

# Define the arrays to store the data
u_pred = np.zeros((steps, 100))
u_target = np.zeros((steps, 100))

# Train the network
for i in range(steps):
    u_target[i, :] = u0 + i * dt * u1
    u_input = np.zeros((100, 2))
    u_input[:, 0] = x
    u_input[:, 1] = u_target[i, :]
    u_input_tensor = torch.tensor(u_input, dtype=torch.float32)
    u_pred_tensor = model(u_input_tensor)
    u_pred[i, :] = u_pred_tensor.detach().numpy().flatten()
    loss = loss_fn(u_pred_tensor, u_input_tensor[:, 1])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot the results
plt.plot(x, u0, 'k', label='IC1')
plt.plot(x, u1, 'b', label='IC2')
plt.plot(x, u_pred[-1, :], 'r', label='Prediction')
plt.legend()
plt.show()
