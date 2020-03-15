import numpy as np


def sigmoid(x, flag=False):
    # flag为true表示要求导，要反向传播
    if (flag == True):
        # 这里是对sigmoid（x）之后的值求导，因为error要先对sigmoid（x）求导，之后再对线形结果求道，接着对对应权重求导
        return x * (1 - x)
    # 前向传播返回值
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    # 五个样本，每个样本有三个特征
    x = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1],
                  [0, 0, 1]])
    # 输出
    y = np.array([[0],
                  [1],
                  [1],
                  [0],
                  [0]])
    # np下random模块的random函数，seed种子确定下来每次的random数值相同。
    np.random.seed(1)
    # 不2乘减1的话是（0，1），希望是（-1，1）
    w0 = 2 * np.random.random((3, 4)) - 1
    w1 = 2 * np.random.random((4, 1)) - 1
    for i in range(60000):
        L0 = x
        # 前向算法
        # 先线形运算
        linear1 = np.dot(x, w0)
        # 激活函数
        L1 = sigmoid(linear1)

        linear2 = np.dot(L1, w1)
        # 前向的最终输出
        L2 = sigmoid(linear2)

        # 假如这里用的loss是均方误差，1/2（y-out)的平方
        # 直接开始反向传播，计算隐层out   要算w1改变对loss的影响  链式求导：loss对sigmoid*sigmoid对linear的相应部分*linear对w
        # 第一步：计算loss对sigmoid后out的导数：2*1/2（y-out）*-1 即 out - y 它就表示真实输出和输出之间的差距error
        L2_error = L2 - y
        if (i % 10000) == 0:
            # loss_out是一个向量
            print("error" + "=" + str(np.mean(np.abs(L2_error))))
        # 第二步，loss对sigmoid 这里乘L2_error是把它当作权重，意思是你错的越多，更新的越大，对应相乘
        # L2_delta是表示你错了多少，这个的结果是sigmoid形式的结果，参数x是L1和w1组成
        L2_delta = L2_error * sigmoid(L2, flag=True)
        # sigmoid对linear的相应部分：它是累加的，所以对相应部分整体求导结果是1
        # 相应部分在对相应权重求导，wx对w求道，这是矩阵求导，记住，结果是x的转置。之后相应w减去累乘结果，就更新完毕。

        # 因为L2 = L1*w1 再经过激活的，所以，L1的error = ΔL2*w1的转置
        L1_error = L2_delta.dot(w1.T)
        L1_delta = L1_error * sigmoid(L1, flag=True)

        w1 -= L1.T.dot(L2_delta)
        w0 -= L0.T.dot(L1_delta)
