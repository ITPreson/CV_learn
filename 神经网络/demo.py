import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

lstm = nn.LSTM(3, 3)  # 输入单词用一个维度为3的向量表示, 隐藏层的一个维度3，仅有一层的神经元，
#记住就是神经元，这个时候神经层的详细结构还没确定，仅仅是说这个网络可以接受[seq_len,batch_size,3]的数据输入
# print(lstm.all_weights)
inputs = [torch.randn(1, 3) for _ in range(5)]
 # 构造一个由5个单单词组成的句子 构造出来的形状是 [5,1,3]也就是明确告诉网络结构我一个句子由5个单词组成，
#每个单词由一个1X3的向量组成，就是这个样子[1,2,3]
#同时确定了网络结构，每个批次只输入一个句子，其中第二维的batch_size很容易迷惑人
#对整个这层来说，是一个批次输入多少个句子，具体但每个神经元，就是一次性喂给神经元多少个单词。

# 初始化隐藏状态
hidden = torch.randn(2, 1, 3)

print(hidden.size(0))