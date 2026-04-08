import os
import torch
import warnings
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils.timefeatures import time_features
from sklearn.preprocessing import StandardScaler
from utils.augmentation import run_augmentation_single


warnings.filterwarnings('ignore') # 忽视所有警告


class SingleDataset(Dataset):
    def __init__(self, data, seq_len, label_len, pred_len, stride):
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.stride = stride
        self.context_len = seq_len + pred_len

    def __getitem__(self, idx):
        s_begin = idx * self.stride
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end, :]
        seq_y = self.data[r_begin:r_end, :]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_y.shape[0], 1))

        return (seq_x, seq_y, seq_x_mark, seq_y_mark)

    def __len__(self):
        total_len = len(self.data)
        n_samples = (total_len - self.context_len) // self.stride + 1
        return n_samples


class Dataset_Pretrain_TimeCAP:
    def __init__(self, args, flag='train', timeenc=0, size=None, scale=True, stride=1, seasonal_patterns=None):
        assert flag in ['train', 'val', 'test', 'inference'], f"flag {flag} is invalid"

        self.seq_len, self.label_len, self.pred_len = size
        self.flag = flag
        self.set_type = {'train': 0, 'val': 1, 'test': 2, 'inference': 2}[flag]

        self.scale = scale
        self.scalers = []  # 不同数据集的标准器
        self.root_path = args.root_path_pretrain
        self.stride = stride
        self.datasets = [] # 不同数据集的Dataset对象

        self.__read_data__()

    def __read_data__(self):
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                # 构造完整路径并读取数据
                if not file.endswith('.csv'):
                    continue
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)

                # 如果第一列是时间戳，则忽略
                if isinstance(df.iloc[0, 0], str):
                    raw_data = df.iloc[:, 1:].values
                else:
                    raw_data = df.values

                # 划分数据集，获得对应数据边界
                if file in ['ETTh1.csv', 'ETTh2.csv']: # 6:2:2
                    border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
                    border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
                elif file in ['ETTm1.csv', 'ETTm2.csv']: # 6:2:2
                    border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
                    border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
                else: # 7:1:2
                    num_train = int(len(raw_data) * 0.7)
                    num_test = int(len(raw_data) * 0.2)
                    num_vali = len(raw_data) - num_train - num_test
                    border1s = [0, num_train - self.seq_len, len(raw_data) - num_test - self.seq_len]
                    border2s = [num_train, num_train + num_vali, len(raw_data)]
                border1 = border1s[self.set_type]
                border2 = border2s[self.set_type]

                # 归一化处理
                if self.scale:
                    scaler = StandardScaler()
                    train_data = raw_data[border1s[0]:border2s[0]]
                    scaler.fit(train_data)
                    raw_data = scaler.transform(raw_data)
                    self.scalers.append(scaler)

                # 提取指定区间数据，判断是否一个样本都没有
                data = raw_data[border1:border2]

                # 构造这一个数据集的Dataset对象
                dataset = SingleDataset(
                    data=data,
                    seq_len=self.seq_len,
                    label_len=self.label_len,
                    pred_len=self.pred_len,
                    stride=self.stride,
                )
                self.datasets.append(dataset)
                print(f"Type: {self.flag}, File: {file}, Number: {len(dataset)}")



class Dataset_ETT_hour(Dataset): # 该类继承自Dataset父类
    def __init__(self, args, flag='train', scale=True, timeenc=0, size=None, freq='h', seasonal_patterns=None):

        # train or test or val
        assert flag in ['train', 'test', 'val', 'inference']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'inference': 2}
        self.set_type = type_map[flag]

        # size [seq_len, label_len, pred_len]
        self.args = args
        self.seq_len = size[0]  # 96
        self.label_len = size[1]  # 48
        self.pred_len = size[2]  # 96
        self.features = args.features
        self.percent = args.percent # 训练样本的比例
        self.target = args.target
        self.scale = scale  # 是否进行数据标准化
        self.timeenc = timeenc
        self.freq = args.freq
        self.root_path = args.root_path
        self.data_path = args.data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler() # StandardScaler对数据进行标准化，使其均值为 0，标准差为 1，确保特征数据的分布一致，避免某些特征因数值较大而主导模型的学习。
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path)) # 将父路径和子路径结合起来返回新的路径，避免windows和Unix平台差异化导致路径出错

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:] # 存储 除第一列外的所有列名
            df_data = df_raw[cols_data] # 去除时间列
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale: # 如果用全部的数据进行标准化的话可能会响标准化参数，导致模型性能虚高。
            train_data = df_data[border1s[0]:border2s[0]] # 提取训练集数据
            self.scaler.fit(train_data.values) # 计算训练集的均值和标准差，.values将pandas提取的数据转变为numpy类型
            data = self.scaler.transform(df_data.values) # 用训练集的均值和方差去标准化整个数据集
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2] # 提取训练数据的时间列
        df_stamp['date'] = pd.to_datetime(df_stamp.date) # 格式转换，便于时间计算、索引、特征提取
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1) # 提取月份信息并存储为一个新的列
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1) # 提取日期信息并存储为一个新的列
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1) # 提取周信息并存储为一个新的列
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1) # 提取小时信息并存储为一个新的列
            data_stamp = df_stamp.drop(['date'], 1).values # 将date列删除，然后将其余的列转换为numpy数组
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2] # 提取训练集或者验证集或者测试集的数据
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp # 时间列表

        if self.set_type == 0 and self.percent != 100: # zero-shot or few-shot样本构造
            num_all = len(self.data_x) - self.seq_len - self.pred_len + 1
            num_few_shot = int(num_all * self.percent / 100)
            index = np.arange(num_all)
            self.index = index[:num_few_shot]

    def __getitem__(self, index):
        if self.set_type == 0 and self.percent != 100:
            index = self.index[index]
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0 and self.percent != 100:
            return len(self.index)
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data) # 反标准化


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, flag='train', scale=True, timeenc=0, size=None, freq='h', seasonal_patterns=None):

        # train or test or val
        assert flag in ['train', 'test', 'val', 'inference']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'inference': 2}
        self.set_type = type_map[flag]

        # size [seq_len, label_len, pred_len]
        self.args = args
        self.seq_len = size[0]  # 96
        self.label_len = size[1]  # 48
        self.pred_len = size[2]  # 96
        self.features = args.features
        self.percent = args.percent  # 训练样本的比例
        self.target = args.target
        self.scale = scale  # 是否进行数据标准化
        self.timeenc = timeenc
        self.freq = args.freq
        self.root_path = args.root_path
        self.data_path = args.data_path
        self.__read_data__()


    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

        if self.set_type == 0 and self.percent != 100: # zero-shot or few-shot样本构造
            num_all = len(self.data_x) - self.seq_len - self.pred_len + 1
            num_few_shot = int(num_all * self.percent / 100)
            index = np.arange(num_all)
            self.index = index[:num_few_shot]

    def __getitem__(self, index):
        if self.set_type == 0 and self.percent != 100:
            index = self.index[index]
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0 and self.percent != 100:
            return len(self.index)
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, flag='train', scale=True, timeenc=0, size=None, freq='h', seasonal_patterns=None):
        # train or test or val
        assert flag in ['train', 'test', 'val', 'inference']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'inference': 2}
        self.set_type = type_map[flag]

        # size [seq_len, label_len, pred_len]
        self.args = args
        self.seq_len = size[0]  # 96
        self.label_len = size[1]  # 48
        self.pred_len = size[2]  # 96
        self.features = args.features
        self.percent = args.percent  # 训练样本的比例
        self.target = args.target
        self.scale = scale  # 是否进行数据标准化
        self.timeenc = timeenc
        self.freq = args.freq
        self.root_path = args.root_path
        self.data_path = args.data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # df_stamp = df_raw[['date']][border1:border2]
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        # if self.timeenc == 0:
        #     df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        #     df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        #     df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        #     df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        #     data_stamp = df_stamp.drop(['date'], 1).values
        # elif self.timeenc == 1:
        #     data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        #     data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        # self.data_stamp = data_stamp

        if self.set_type == 0 and self.percent != 100: # zero-shot or few-shot样本构造
            num_all = len(self.data_x) - self.seq_len - self.pred_len + 1
            num_few_shot = int(num_all * self.percent / 100)
            index = np.arange(num_all)
            self.index = index[:num_few_shot]

    def __getitem__(self, index):
        if self.set_type == 0 and self.percent != 100:
            index = self.index[index]
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0 and self.percent != 100:
            return len(self.index)
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, args, flag='train', scale=True, timeenc=0, size=None, freq='h', seasonal_patterns=None):
        # train or test or val
        assert flag in ['train', 'test', 'val', 'inference']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'inference': 2}
        self.set_type = type_map[flag]

        # size [seq_len, label_len, pred_len]
        self.args = args
        self.seq_len = size[0]  # 96
        self.label_len = size[1]  # 48
        self.pred_len = size[2]  # 96
        self.features = args.features
        self.percent = args.percent  # 训练样本的比例
        self.target = args.target
        self.scale = scale  # 是否进行数据标准化
        self.timeenc = timeenc
        self.freq = args.freq
        self.root_path = args.root_path
        self.data_path = args.data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.percent != 100: # zero-shot or few-shot样本构造
            num_all = len(self.data_x) - self.seq_len - self.pred_len + 1
            num_few_shot = int(num_all * self.percent / 100)
            index = np.arange(num_all)
            self.index = index[:num_few_shot]

    def __getitem__(self, index):
        if self.set_type == 0 and self.percent != 100:
            index = self.index[index]
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0 and self.percent != 100:
            return len(self.index)
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
