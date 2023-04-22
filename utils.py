import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import os
from sklearn import preprocessing
import math
import torch.nn as nn
from model import ResNet
from tent import softmax_entropy


class DatasetOfDiv(Dataset):
    def __init__(self, data_features, data_target):
        self.len = len(data_features)
        self.features = torch.from_numpy(data_features).float()
        self.target = torch.from_numpy(data_target).long()

    def __getitem__(self, index):
        return self.features[index], self.target[index]

    def __len__(self):
        return self.len

def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res

def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path

def getData(dataset_name):
    root_dir_dataset = './dataset/' + dataset_name  # 存放测试数据
    df_train = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TRAIN.tsv', sep='\t', header=None)
    df_test = pd.read_csv(root_dir_dataset + '/' + dataset_name + '_TEST.tsv', sep='\t', header=None)

    y_train = df_train.values[:, 0]
    y_test = df_test.values[:, 0]

    x_train = df_train.drop(columns=[0])
    x_test = df_test.drop(columns=[0])

    x_train.columns = range(x_train.shape[1])
    x_test.columns = range(x_test.shape[1])

    x_train = x_train.values  # Return a Numpy representation of the DataFrame.
    x_test = x_test.values

    # znorm
    std_ = x_train.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

    std_ = x_test.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))  # 转换成二维数组 生成每种label的独热向量
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()  # y_train用独热向量表示
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # 得到从0开始的标签
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)  # Returns the indices of the maximum values along an axis.

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

    return x_train, x_test, y_train, y_test, nb_classes

def getModel(dataset_name2, nb_classes):
    best_model_directory = './best_model/' + dataset_name2 + '/'
    # 加载模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # create model
    model = ResNet(classes=nb_classes).to(device)
    # load model weights
    weights_path = best_model_directory + 'best_model.pth'
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    # model.load_state_dict(torch.load(weights_path))  # GPU的时候
    model.load_state_dict(torch.load(weights_path,map_location=torch.device('cpu')))  # cpu的时候
    return model

def clean_accuracy(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   batch_size: int = 16,
                   device: torch.device = None,
                   output_directory = None,
                   dataset_name2 = None):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size) # 向上舍入到最接近的整数
    acc_curr_list=[]
    entropy_list=[]
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) *
                       batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) *
                       batch_size].to(device)

            output = model(x_curr)
            entropy = softmax_entropy(output).mean(0)
            acc_curr = (output.max(1)[1] == y_curr).float().sum()
            acc += (output.max(1)[1] == y_curr).float().sum()
            acc_curr_list.append(acc_curr.item() / x_curr.shape[0])
            entropy_list.append(entropy.item())
            #acc_curr_list.append((acc_curr / x_curr.shape[0]).item().float())
            print('batch:[{}/{}], accuracy: {}, entropy: {}'.format(counter+1, n_batches,
                                                                    acc_curr.item() / x_curr.shape[0], entropy.item()))

        # 每个batch的accuracy
        create_directory(output_directory)
        acc_curr_list = pd.DataFrame(acc_curr_list)
        ent_mean = np.mean(entropy_list)
        entropy_list = pd.DataFrame(entropy_list)
        save_path = output_directory + 'acc_curr_list_' + dataset_name2 + '.csv'
        acc_curr_list.to_csv(save_path, index=False)
        save_path2 = output_directory + 'entropy_list_' + dataset_name2 + '.csv'
        entropy_list.to_csv(save_path2, index=False)


    return acc.item() / x.shape[0], ent_mean