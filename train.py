import torch
import torch.nn as nn
from model import ResNet
from torch.utils.data import DataLoader
import time
import torch.optim as optim
from utils import DatasetOfDiv
from utils import getData
import sys
from utils import create_directory
from dataloader import data_generator
import argparse


def train():
    best_model_directory = './best_model/' + dataset_name + '/'
    # 创建目录
    create_directory(best_model_directory)
    # 训练参数
    learning_rate = 0.001
    num_epochs = 50
    best_acc = 0.0
    min_loss = 10000
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = best_model_directory+'best_model.pth'
    batchSize = 64
    bestepoch=0

    if dataset_name == "EEG":
        best_model_directory = best_model_directory + src_id + '/'
        create_directory(best_model_directory)
        save_path = best_model_directory + 'best_model.pth'
        train_loader, test_loader = data_generator(dataset_name, src_id)
        nb_classes = 5
    else:
        # 加载数据
        x_train, x_test, y_train, y_test, nb_classes = getData(dataset_name)
        trainSet=DatasetOfDiv(x_train, y_train)
        train_loader = DataLoader(dataset=trainSet, batch_size=batchSize, shuffle=True)
        testSet=DatasetOfDiv(x_test, y_test)
        test_loader = DataLoader(dataset=testSet, batch_size=batchSize, shuffle=False)

    model = ResNet(in_channels=1, classes=nb_classes)
    model.to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 定义学习率策略
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True, threshold=0.0001, threshold_mode='rel',
        cooldown=0, min_lr=0, eps=1e-10)

    # Train the model
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss=0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            # Backward and optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step(loss)  # scheduler为学习率调整策略,针对loss进行学习率改变。
            # 记得加上评价指标loss。这条语句可放在epoch的循环位置，要放在batch循环位置也可以，只是正对patience对象不同。

        if (epoch+1) % 10 == 0:
            print('Epoch: [{}/{}], Loss: {}'.format(epoch + 1, num_epochs, epoch_loss))

        # Test the model
        model.eval()  # eval mode(batch norm uses moving mean/variance
        # instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            val_loss = 0
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += loss.item()
        val_acc = correct / total
        if (epoch+1) % 10 == 0:
            print('Test accuracy of the model on the test data: {} %'.format(100 * correct / total))
            print('Loss of the model on the test data: {}'.format(val_loss))
        #print('Test accuracy of the model on the test data: {} %'.format(100 * correct / total))
        if val_acc>=best_acc and val_loss<min_loss:
            best_acc=val_acc
            min_loss=val_loss
            torch.save(model.state_dict(), save_path)
            bestepoch=epoch

    duration = time.time() - start_time
    print('best epoch:{}, best_acc:{}, min_loss:{}'.format(bestepoch+1,best_acc,min_loss))
    print('Finished Training')

# this is the code used to launch an experiment on a dataset

# 训练
if __name__ == '__main__':
    # dataset_name = sys.argv[1]
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('--data', default='EEG', help='training dataset')
    parser.add_argument("--scrID", default='0', help='domain_id')
    args = parser.parse_args()
    dataset_name = args.data
    src_id = args.scrID
    train()