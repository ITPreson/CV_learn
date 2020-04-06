import numpy as np
import torch
import torch.nn as nn


# 用numpy实现一个简单神经网络

def N_fromNumpy():
    # hyper refer
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10
    # Create random input and output data
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)
    # Randomly initialize weights
    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)

    # learning rate
    learning_rate = 1e-6

    for i in range(500):
        # foreward pass
        h = x.dot(w1)
        h_relu = np.maximum(0, h)
        y_pred = h_relu.dot(w2)

        # loss 这里用均方误差，只不过没有除64而已，好算梯度
        loss = np.square(y_pred - y).sum()
        print(i, loss)

        # backward pass  记住主要是算权重参数
        grad_y_pred = 2 * (y_pred - y)
        # grad_w2 = grad_y_pred.dot(h_relu.T)  # 这样不行
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        # h小于0的导数等于0，h大于0的导数等于1，再乘之前算的导数，相当于不变，所以copy后，h<0置0就好
        grad_h[h < 0] = 0
        # grad_w1 = grad_h.dot(x.T)
        grad_w1 = x.T.dot(grad_h)

        # update weight
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2


def N_fromTorch():
    # hyper refer
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10
    # Create random input and output data
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)
    # Randomly initialize weights
    w1 = torch.randn(D_in, H)
    w2 = torch.randn(H, D_out)

    # learning rate
    learning_rate = 1e-6

    for i in range(500):
        # foreward pass
        h = x.mm(w1)  # matrix multi
        h_relu = h.clamp(min=0)  # 挤压到一个范围
        y_pred = h_relu.mm(w2)

        # loss 这里用均方误差，只不过没有除64而已，好算梯度
        loss = (y_pred - y).pow(2).sum().item()
        print(i, loss)

        # backward pass  记住主要是算权重参数
        grad_y_pred = 2.0 * (y_pred - y)
        # grad_w2 = grad_y_pred.dot(h_relu.T)  # 这样不行
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        # h小于0的导数等于0，h大于0的导数等于1，再乘之前算的导数，相当于不变，所以copy后，h<0置0就好
        grad_h[h < 0] = 0
        # grad_w1 = grad_h.dot(x.T)
        grad_w1 = x.t().mm(grad_h)

        # update weight
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2


def N_fromTorch_moreEasy():
    # hyper refer
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10
    # Create random input and output data
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    # learning rate

    model = nn.Sequential(

        nn.Linear(D_in, H),
        nn.ReLU(),
        nn.Linear(H, D_out),

    )

    # 这里是你觉得自动初始化的不好，你自己初始化。
    # nn.init.normal_(model[0].weight)
    # nn.init.normal_(model[2].weight)

    loss_fn = nn.MSELoss(reduction='sum')
    learning_rate = 1e-4  # Adam一般learning_rate是 -3 -4   SGD -5到-6
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    for i in range(500):
        # foreward pass

        y_pred = model(x)

        # loss 这里用均方误差，只不过没有除64而已，好算梯度
        loss = loss_fn(y_pred, y)
        print(i, loss.item())

        # 算梯度前先清零
        optimizer.zero_grad()

        # backward pass  记住主要是算权重参数
        loss.backward()

        # 算梯度后，一步更新梯度
        optimizer.step()

        # update weight 这样写还是麻烦，直接用现成的优化器
        # with torch.no_grad():
        #     for param in model.parameters():
        #         param -= learning_rate * param.grad
        # model.zero_grad()


def N_fromTorch_standard():

    '''
        虽然形式上复杂了，但对最后写大型模型有利，记住这种写法
    :return:
    '''
    class TwoLayerNet(nn.Module):
        def __init__(self, D_in, H, D_out):
            # 这里面定义模型的框架
            super(TwoLayerNet, self).__init__()
            self.linear1 = nn.Linear(D_in, H, bias=False)
            self.linear2 = nn.Linear(H, D_out, bias=False)

        def forward(self, x):
            y_pred = self.linear2(self.linear1(x).clamp(min=0))
            return y_pred

    # hyper refer
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10


    # Create random input and output data
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    # learning rate

    model = TwoLayerNet(D_in, H, D_out)

    # 这里是你觉得自动初始化的不好，你自己初始化。
    # nn.init.normal_(model[0].weight)
    # nn.init.normal_(model[2].weight)

    loss_fn = nn.MSELoss(reduction='sum')
    learning_rate = 1e-4  # Adam一般learning_rate是 -3 -4   SGD -5到-6
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    for i in range(500):
        # foreward pass

        y_pred = model(x)

        # loss 这里用均方误差，只不过没有除64而已，好算梯度
        loss = loss_fn(y_pred, y)
        print(i, loss.item())

        # 算梯度前先清零
        optimizer.zero_grad()

        # backward pass  记住主要是算权重参数
        loss.backward()

        # 算梯度后，一步更新梯度
        optimizer.step()


if __name__ == '__main__':
    # N_fromNumpy()
    # N_fromTorch()
    # N_fromTorch_moreEasy()
    N_fromTorch_standard()
