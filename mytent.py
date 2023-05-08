import pandas as pd

from model import ResNet
import tent
import torch
import torch.optim as optim
from utils import getData
from utils import clean_accuracy as accuracy
from utils import clean_accuracy_new as accuracy_new
from utils import getModel
from utils import getModel_new
from utils import create_directory
import sys
import argparse
from dataloader import data_generator

def mytent():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if dataset_name == "EEG":
        # 加载模型
        nb_classes = 5
        model = getModel_new(dataset_name, src_id, nb_classes)
        if method == "tent":
            # tent模型
            tented_model = setup_tent(model)
        if method == "source":
            tented_model = setup_source(model)
        train_loader, test_loader = data_generator(dataset_name, trg_id, batchsize=64)
        acc, ent_mean = accuracy_new(tented_model, test_loader, device=device)
        print('accuracy: {}, mean_entropy: {}'.format(acc, ent_mean))

        output_directory = './results/' + dataset_name + '/' + trg_id + '/'  # 结果
        create_directory(output_directory)
        save_path = output_directory + method + '_acc_' + src_id + '.csv'
        data = {'acc': [acc], 'ent': [ent_mean]}
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)

    else:
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

        output_directory = './results/' + dataset_name + '/'  # 结果
        create_directory(output_directory)
        acc, ent_mean = accuracy(tented_model, Xtest, Ytest, device=device, output_directory=output_directory,
                       dataset_name2=dataset_name2)
        print('accuracy: {}, mean_entropy: {}'.format(acc, ent_mean))
        save_path = output_directory + method + 'acc_' + dataset_name2 + '.csv'
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
    # dataset_name = sys.argv[1]  # 测试集
    # dataset_name2 = sys.argv[2]  # 最佳模型
    # steps = int(sys.argv[3])
    # method = sys.argv[4]

    # output_directory = './results/' + dataset_name + '/' # 结果
    # mytent()

    parser = argparse.ArgumentParser(description="train")
    parser.add_argument('--data', default='EEG', help='testing dataset')
    parser.add_argument('--Sdata', default='EEG', help='training dataset')
    parser.add_argument("--scrID", default='0', help='scr_id')
    parser.add_argument("--trgID", default='0', help='trg_id')
    parser.add_argument("--steps", type=int, default='1', help='steps of tent')
    parser.add_argument("--method", default='source', help='method: tent or source')
    args = parser.parse_args()
    dataset_name = args.data
    dataset_name2 = args.Sdata
    src_id = args.scrID
    trg_id = args.trgID
    steps = args.steps
    method = args.method

    output_directory = './results/' + dataset_name + '/'  # 结果
    create_directory(output_directory)
    mytent()