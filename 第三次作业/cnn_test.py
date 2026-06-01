import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from torchvision.datasets import ImageFolder

IMAGE_SIZE = 64
NUM_CLASSES = 3
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

TRAIN_DATA_PATH = r"D:\下载\DL\DL\dataset\train"
TEST_DATA_PATH = {
    "test1": r"D:\下载\DL\DL\dataset\test1",
    "test2": r"D:\下载\DL\DL\dataset\test2"
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(root=TRAIN_DATA_PATH, transform=transform)

print(f"训练集总图片数: {len(train_dataset)}")
print(f"类别列表: {train_dataset.classes}")
print(f"类别与标签对应: {train_dataset.class_to_idx}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层1：输入3通道，输出16通道，3×3卷积核
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)
        # 卷积层2：输入16通道，输出32通道
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        # 卷积层3：输入32通道，输出64通道
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        # 池化层：2×2最大池化，步长2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU()

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 64, 64)
            dummy_output = self.pool(self.conv1(dummy_input))
            dummy_output = self.pool(self.conv2(dummy_output))
            dummy_output = self.pool(self.conv3(dummy_output))
            fc_input_size = dummy_output.numel()

        # 全连接层1(输入自动计算，输出128)
        self.fc1 = nn.Linear(fc_input_size, 128)
        # 全连接层2(输入128，输出64)
        self.fc2 = nn.Linear(128, 64)
        # 全连接层3(输入64，输出3)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def test_model(test_path):
    # 加载当前测试集
    test_dataset = ImageFolder(root=test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model.eval()  # 切换到评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 关闭梯度计算
        for images, labels in test_loader:
            # 移动数据
            images = images.to(device)
            labels = labels.to(device)
            # 前向传播
            outputs = model(images)
            # 取概率最大的那个类别作为预测结果
            _, predicted = torch.max(outputs.data, 1)
            # 统计总样本数
            total += labels.size(0)
            # 统计预测正确的样本数
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"  测试集总图片数: {len(test_dataset)}")
    print(f"  准确率: {accuracy:.2f}%")
    return accuracy


total_step = len(train_loader)
train_losses = []
test_accuracies = []

for epoch in range(NUM_EPOCHS):
    model.train()  # 切换到训练模式
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        # 将数据移动到GPU/CPU
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

    avg_train_loss = running_loss / total_step
    train_losses.append(avg_train_loss)

    acc_epoch = test_model(TEST_DATA_PATH["test1"])
    test_accuracies.append(acc_epoch)

    print(f'第 {epoch + 1} 轮结束 | 平均训练损失: {avg_train_loss:.4f} | test1监控准确率: {acc_epoch:.2f}%\n')

print("\n测试test1数据集")
acc1 = test_model(TEST_DATA_PATH["test1"])

print("\n测试test2数据集")
acc2 = test_model(TEST_DATA_PATH["test2"])

print(f"test1 准确率: {acc1:.2f}%")
print(f"test2 准确率: {acc2:.2f}%")

torch.save(model.state_dict(), 'cone_classifier.pth')
print("\n✅ 模型已保存为 cone_classifier.pth")