import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

# Hyper parameter
input_size = 1
time_step = 10
learning_rate = 0.02

# steps = np.linspace(0, np.pi * 2, 100, dtype=np.float32)  # float32 for converting torch FloatTensor
# x_np = np.sin(steps)
# y_np = np.cos(steps)


# plt.plot(steps, y_np, 'r-', label='target (cos)')
# plt.plot(steps, x_np, 'b-', label='input (sin)')
# plt.legend(loc='best')
# plt.show()
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        x, h_state = self.rnn(x, h_state)
        outs = []  # # calculate output for each time step
        for timeStep in range(x.size(1)):
            outs.append(self.fc(x[:, timeStep, :]))
        return torch.stack(outs, dim=1), h_state


rnn = RNN()

optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)  # optimize all cnn parameters
loss_func = nn.MSELoss()

h_state = None  # for initial hidden state

plt.figure(1, figsize=(12, 5))
plt.ion()
plt.show()

for step in range(100):
    start, end = step * np.pi, (step+1)*np.pi   # time range
    # use sin predicts cos
    steps = np.linspace(start, end, time_step, dtype=np.float32, endpoint=False)  # float32 for converting torch FloatTensor
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])    # shape (batch, time_step, input_size)
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction, h_state = rnn(x, h_state)   # rnn output
    # !! next step is important !!
    h_state = h_state.data        # repack the hidden state, break the connection from last iteration

    loss = loss_func(prediction, y)         # calculate loss
    optimizer.zero_grad()                   # clear gradients for this training step
    loss.backward()                         # backpropagation, compute gradients
    optimizer.step()                        # apply gradients

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.05)

