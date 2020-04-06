import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Hyper parameters
num_epoch = 1
batch_size = 64
input_size = 28
time_step = 28
learning_rate = 0.01

train_dataSets = torchvision.datasets.MNIST(root='./', train=True, transform=transforms.ToTensor(), download=False)
test_dataSets = torchvision.datasets.MNIST(root='./', train=False, transform=transforms.ToTensor())
train_dataLoader = DataLoader(dataset=train_dataSets, batch_size=batch_size, shuffle=True)
test_dataLoader = DataLoader(dataset=test_dataSets, batch_size=batch_size, shuffle=True)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        x, (h_n, h_c) = self.lstm(x, None)  # None represents zero initial hidden state
        # choose r_out at the last time step
        x = self.fc(x[:, -1, :])
        return x


rnn = RNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

# train
for epoch in range(num_epoch):
    for step, (images, labels) in enumerate(train_dataLoader):
        images = images.view(-1, 28, 28)  # reshape x to (batch, time_step, input_size)
        out = rnn(images)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())
# test
corr = 0
total = 0
with torch.no_grad():
    for images, labels in test_dataLoader:
        images = images.view(-1, 28, 28)
        out = rnn(images)
        total += labels.size(0)
        _, predict = torch.max(out.data, 1)
        corr += (predict == labels).sum().item()

print('Test Accuracy of the model on the 10000 test images: {}%'
      .format(100 * corr / total))
