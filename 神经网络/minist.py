import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# hyper parameter
input_size = 28 * 28
hidden_size = 500
num_classes = 10
learning_rate = 0.001
batch_size = 100
num_epoch = 5

train_dataset = torchvision.datasets.MNIST(root='./',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=False)
test_dataset = torchvision.datasets.MNIST(root='./',
                                          train=False,
                                          transform=transforms.ToTensor())
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)


class net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output = self.fc1(x)
        self.relu()
        output = self.fc2(output)
        return output


model = net(input_size, hidden_size, num_classes)

# loss_fu optimizer
loss_fu = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train
total_step = len(train_dataloader)
for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_dataloader):

        images = images.reshape(-1, 28 * 28)

        # forward pass
        out = model(images)
        loss = loss_fu(out, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
           print('Epoch [{}/{}],step [{}/{}],loss {:.4f}'
                 .format(epoch+1,num_epoch,i+1,total_step,loss.item()))

# test
with torch.no_grad():
    total = 0
    correct = 0
    for images,labels in test_dataloader:
        images = images.reshape(-1, 28 * 28)
        out = model(images)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    # Save the model checkpoint

torch.save(model.state_dict(), 'weight/model.pth')


