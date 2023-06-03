import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import time

# 记录开始时间
start_time = time.time()
# 超参数
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True  # True：还没有下载    False：下载好了

# 下载和加载MNIST训练数据集
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

# 创建训练数据集的数据加载器
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# 下载和加载MNIST测试数据集
test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.float32)[:2000] / 255.
test_y = test_data.targets[:2000]


# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # (1, 28, 28)
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,  # 如果stride = 1，padding = (kernel_size - 1)/2 = (5-1)/2
            ),  # 卷积层（就是过滤器，filter） # -> (16, 28, 28)
            nn.ReLU(),  # 激活函数（神经网络）    # -> (16, 28, 28)
            nn.MaxPool2d(kernel_size=2),  # 池化层 # -> (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # -> (32, 14, 14)
            nn.ReLU(),  # -> (32, 14, 14)
            nn.MaxPool2d(2)  # -> (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1)  # (batch, 32 * 7 * 7)
        output = self.out(x)
        return output


if __name__ == '__main__':
    import numpy as np

    cnn = CNN()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # 优化所有CNN参数
    loss_func = nn.CrossEntropyLoss()  # 目标标签不是one-hot编码

    # 训练和测试
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):

            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_output = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = (sum(pred_y == np.array(test_y.data)).item()) / test_y.size(0)
                print('Epoch:', epoch, '| train loss:%.4f' % loss.item(), '| test accuracy:%.4f' % accuracy)
                # 打印测试数据集的前10个预测结果

    test_output = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, '预测数字')
    print(test_y[:10].numpy(), '真实数字')

    # 记录结束时间
    end_time = time.time()

    # 计算整个运行时间
    runtime = end_time - start_time
    print('总运行时间为：%.2f秒' % runtime)
