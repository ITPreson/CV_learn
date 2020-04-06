import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# Hyper-parameter
num_epoch = 5
batch_size = 100
learning_rate = 0.001
num_classes = 10

train_dataset = torchvision.datasets.MNIST(root="./",
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=False)
test_dataset = torchvision.datasets.MNIST(root='./',
                                          train=False,
                                          transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


class ConvNet(nn.Module):
    # define two convolution layer
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # fully connection
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        out = self.fc(out)
        return out


model = ConvNet(num_classes)

# define loss function and optimizer
loss_fuc = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        # forward pass
        out = model(images)
        loss = loss_fuc(out, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print("Epoch [{}/{}],step [{}/{}],loss{:.4f}"
                  .format(epoch + 1, num_epoch, i + 1, total_step, loss.item()))

# test
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        out = model(images)
        total += labels.size(0)
        _, predict = torch.max(out.data, 1)  # 这里的dim=1 代表返回行上的最大值及其位置索引
        correct += (labels == predict).sum().item()

print('Test Accuracy of the model on the 10000 test images: {}%'
      .format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'weight/conv_model.pth')
