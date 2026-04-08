import os
import math
import time
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections.abc import Iterable

plt.switch_backend('agg')

# 定义随机种子固定的函数
def set_seed(fix_seed):
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(fix_seed)


# 检查路径是否存在
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_dir(args):

    model_result_dir = os.path.join(args.res_dir, args.model)
    check_dir(model_result_dir)

    # 将两个字符串连接形成一个新的路径
    test_dir = os.path.join(model_result_dir, 'test') # 存储预测结果，指标
    check_dir(test_dir)

    model_dir = os.path.join(model_result_dir, 'model') # 存储模型权重
    check_dir(model_dir)

    log_dir = os.path.join(model_result_dir, 'log') # 日志文件
    check_dir(log_dir)

    return test_dir, model_dir, log_dir

# 代码运行产生日志文件
def init_logger(log_dir):

    # level: 日志记录的级别为 INFO, 普通信息，error等都可以被记录
    # format: 定义日志输出的格式，每一条日志信息包括时间和信息
    # datefmt: 设置时间戳的格式
    # filename: 设置日志文件的路径和名称, 本次实验的日志都记录在当前日志文件中，一天跑的实验都在一个日志文件里
    # filemode: 文件打开模式为追加模式

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s   %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(log_dir, time.strftime("%Y_%m_%d") + '.log'),
                        filemode='a')

    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # console.setFormatter(formatter)
    # logging.getLogger('').addHandler(console)

    return logging

# 更改学习器的学习率
def adjust_learning_rate(optimizer, epoch, args, scheduler=None):
    if args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate} # 不衰减
    elif args.lradj == "decay":
        lr_adjust = {epoch: args.learning_rate * (args.lr_decay ** (epoch - 1  // 1)) } # TimeXer数据: 跑完第二个epoch后开始衰减 (epoch - 1)
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))} # Cosine Annealing 余弦退火调整
    elif args.lradj == "piecewise":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8} # piecewise 分段调整
    elif args.lradj == 'decay_3_0.8':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.8 ** ((epoch - 2) // 1))} # 跑完3个epoch后学习率衰减20%
    elif args.lradj == "decay_3_0.9":
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 2) // 1))} # 跑完3个epoch后学习率衰减10%
    elif args.lradj == "decay_5_0.1":
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1} # 跑完5个epoch后学习率衰减90%
    elif args.lradj == "decay_10_0.1":
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1} # 跑完10个epoch后学习率衰减90%
    elif args.lradj == "decay_15_0.1":
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1} # 跑完15个epoch后学习率衰减90%
    elif args.lradj == "decay_25_0.1":
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1} # 跑完25个epoch后学习率衰减90%

    # 使用调度器
    elif args.lradj == "use_scheduler":
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        tqdm.write('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.test_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, test_loss, model, path, model_name, epoch, task_name='pretrain', logger=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, test_loss, model, path, logger, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, test_loss, model, path, logger, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, test_loss, model, path, logger, epoch):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model of epoch {epoch}')
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model of epoch {epoch}')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
        self.test_loss_min = test_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def transfer_weights(weights_path, model, exclude_head=True, device='cpu'):
    new_state_dict = torch.load(weights_path,  map_location=device)['model_state_dict'] # 加载预训练权重字典

    matched_layers = 0
    unmatched_layers = []
    for name, param in model.state_dict().items(): # 遍历微调模型的模块
        if exclude_head and 'head' in name: continue # 跳过预测头（TimeDART中预训练和微调的头不一样，避免微调头加载预训练头的权重）
        if name in new_state_dict:
            matched_layers += 1
            input_param = new_state_dict[name]
            if input_param.shape == param.shape:
                param.copy_(input_param)
            else:
                unmatched_layers.append(name)
        else:
            unmatched_layers.append(name)
            pass # these are weights that weren't in the original model, such as a new head
    if matched_layers == 0:
        raise Exception("No shared weight names were found between the models")
    else:
        if len(unmatched_layers) > 0:
            print(f'check unmatched_layers: {unmatched_layers}')
        else:
            print(f"weights from {weights_path} successfully transferred!\n")
    model = model.to(device)
    return model


def show_series(batch_x, batch_x_m, pred_batch_x, idx, time_points=336):

    batch_x = batch_x.permute(0, 2, 1).reshape(batch_x.shape[0], -1)
    batch_x_m = batch_x_m.permute(0, 2, 1).reshape(batch_x_m.shape[0], -1)
    pred_batch_x = pred_batch_x.permute(0, 2, 1).reshape(batch_x_m.shape[0], -1)

    bs = batch_x.shape[0]

    if time_points is None:
        time_points = batch_x.shape[1]

    positive_numbers = batch_x_m.shape[0] // bs

    batch_x = batch_x.numpy()
    batch_x_m = batch_x_m.numpy()

    x = list(range(time_points))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'b']

    fig, axs = plt.subplots(2, 1, figsize=(20, 20))

    for t_i in range(time_points - 1):
        for pn in range(positive_numbers):
            s_i = pn * bs + idx
            if batch_x_m[s_i][t_i] == 0:
                axs[0].plot([x[t_i], x[t_i + 1]], [batch_x[idx][t_i], batch_x[idx][t_i + 1]], ':', color='grey', alpha=0.5, label='masked')
            else:
                axs[0].plot([x[t_i], x[t_i + 1]], [batch_x[idx][t_i], batch_x[idx][t_i + 1]], '-', color=colors[pn], label='unmasked')

        axs[1].plot([x[t_i], x[t_i + 1]], [batch_x[idx][t_i], batch_x[idx][t_i + 1]], '-', color='blue', label='original')
        axs[1].plot([x[t_i], x[t_i + 1]], [pred_batch_x[idx][t_i], pred_batch_x[idx][t_i + 1]], '-', color='orange', label='prediction')

    axs[0].set_title('Multi-masked time series')
    axs[0].set_xlabel('X - time points')
    axs[0].set_ylabel('Y - time values')

    axs[1].set_title('Original vs Reconstruction')
    axs[1].set_xlabel('X - time points')
    axs[1].set_ylabel('Y - time values')

    return fig


def clever_format(nums, format="%.2f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)

    return clever_nums