import os
import numpy as np
import sys
import torch
from utils import calculate_metrics
from utils import getData
from utils import getModel

def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载数据
    x_train, x_test, y_train, y_test, nb_classes = getData(dataset_name)
    # 加载模型
    model = getModel(dataset_name2, nb_classes)
    # 测试模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        Xtest = torch.from_numpy(x_test).float().to(device)
        Ytest = torch.from_numpy(y_test).long().to(device)
        outputs = model(Xtest)
        _, predicted = torch.max(outputs.data, 1)
        total += Ytest.size(0)
        correct += (predicted == Ytest).sum().item()
        print('Test accuracy of the model on the test data: {} %'.format(100 * correct / total))
        y_pred = outputs.cpu().numpy()  # 返回测试集预测概率
        # save predictions
        pred_name = 'y_pred_' + dataset_name2 + '.npy'
        np.save(output_directory + pred_name, y_pred)
        y_pred = predicted.cpu().numpy()

        duration = 0.1919  # 随便写的
        df_metrics = calculate_metrics(y_test, y_pred, duration)
        metrics_name = 'df_metrics_' + dataset_name2 + '.csv'
        df_metrics.to_csv(output_directory + metrics_name, index=False)

# 调用模型对指定测试集进行测试
if __name__ == '__main__':
    dataset_name = sys.argv[1]  # 测试集
    dataset_name2 = sys.argv[2]  # 最佳模型
    output_directory = './results/' + dataset_name + '/'
    test()