import numpy as np
import torch


def simple_imple():
    def encode(input):
        if input % 15 == 0:
            return 3
        elif input % 5 == 0:
            return 2
        elif input % 3 == 0:
            return 1
        else:
            return 0

    def decode(input):
        return [str(input), 'Fizz', "buzz", "FizzBuzz"][encode(input)]

    for i in range(1, 16):
        temp = decode(i)
        print(temp)


def Learn_game():
    def encode(input):
        if input % 15 == 0:
            return 3
        elif input % 5 == 0:
            return 2
        elif input % 3 == 0:
            return 1
        else:
            return 0

    def decode(i,x):
        return [str(i), 'Fizz', "buzz", "FizzBuzz"][x]

    NUM_DIGITS = 10

    def binary_encode(i, num_digits):
        return np.array([i >> d & 1 for d in range(num_digits)])

    trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
    # 输出是一个表示类别的信息，所以要是一个Integer  long = int64
    trY = torch.LongTensor([encode(i) for i in range(101, 2 ** NUM_DIGITS)])

    NUM_HIDDEN = 100
    model = torch.nn.Sequential(
        torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(NUM_HIDDEN, 4)  # 4 logits,after softMax ,we get a probability distribution
    )

    # 上节课是一个回归拟合问题，现在是一个分类问题，一般用交叉熵损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    BATCH_SIZE = 128
    for epoch in range(10000):
        for start in range(0, len(trX), BATCH_SIZE):
            end = start + BATCH_SIZE
            batchX = trX[start:end]
            batchY = trY[start:end]

            y_pred = model(batchX)

            loss = loss_fn(y_pred, batchY)

            print("epoch", epoch, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # 保存训练好的模型
    torch.save(model.state_dict(),'/Users/wang/PycharmProjects/opencv_practice/pytorch/weight/model_para.pth')

def test_model():
    NUM_DIGITS = 10
    def decode(i,x):
        return [str(i), 'Fizz', "buzz", "FizzBuzz"][x]

    def binary_encode(i, num_digits):
        return np.array([i >> d & 1 for d in range(num_digits)])

    NUM_HIDDEN = 100
    model = torch.nn.Sequential(
        torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(NUM_HIDDEN, 4)  # 4 logits,after softMax ,we get a probability distribution
    )
    checkpoint = torch.load('/Users/wang/PycharmProjects/opencv_practice/pytorch/weight/model_para.pth')
    model.load_state_dict(checkpoint)
    testX = torch.Tensor([binary_encode(i,NUM_DIGITS)for i in range(1,101)])
    # 测试的时候就不要梯度了，以免内存爆炸
    with torch.no_grad():
        testY = model(testX)
    # testY.max(dim=1)[1] 维度都是从0开始，max(1)表示返回宽中最大的数值，及对应的第几个位置。[1]表示第几个位置，argMax
    predictions = zip(range(0,101),testY.max(dim=1)[1].tolist())
    print([decode(i+1,x)for i,x in predictions])

if __name__ == '__main__':
    # simple_imple()
    # Learn_game()
    # test_model()
    print([1, 2] + [3, 4])