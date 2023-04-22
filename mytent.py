import pandas as pd

from model import ResNet
import tent
import torch
import torch.optim as optim
from utils import getData
from utils import clean_accuracy as accuracy
from utils import getModel
import sys

def mytent():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载数据
    x_train, x_test, y_train, y_test, nb_classes = getData(dataset_name)
    # 加载模型
    model = getModel(dataset_name2, nb_classes)
    if method == "tent":
        # tent模型
        tented_model = setup_tent(model)
    if method == "source":
        tented_model = setup_source(model)

    # 训练tented_model
    Xtest = torch.from_numpy(x_test).float()
    Ytest = torch.from_numpy(y_test).long()

    acc, ent_mean = accuracy(tented_model, Xtest, Ytest, device=device, output_directory=output_directory,
                   dataset_name2=dataset_name2)
    print('accuracy: {}, mean_entropy: {}'.format(acc, ent_mean))
    save_path = output_directory + 'acc_' + dataset_name2 + '.csv'
    acc = pd.DataFrame([acc], columns=['acc'])
    acc.to_csv(save_path, index=False)

def setup_tent(model):
    # tent模型
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = optim.Adam(params, lr=1e-3)
    tented_model = tent.Tent(model, optimizer, steps=steps)
    return tented_model

def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    return model

# 使用tent算法更新模型
if __name__ == '__main__':
    dataset_name = sys.argv[1]  # 测试集
    dataset_name2 = sys.argv[2]  # 最佳模型
    steps = int(sys.argv[3])
    method = sys.argv[4]

    output_directory = './results/' + dataset_name + '/' # 结果
    mytent()