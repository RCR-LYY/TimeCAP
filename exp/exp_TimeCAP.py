import os
import sys
import time
import torch
import warnings
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch import optim
from utils.metrics import metric
from exp.exp_basic import Exp_Basic
from optimizer.Stable_Spam import StableSPAM
from data_provider.data_factory_TimeCAP import data_provider
from utils.tools import check_dir, EarlyStopping, adjust_learning_rate, clever_format

warnings.filterwarnings('ignore')

class Exp_TimeCAP(Exp_Basic):
    def __init__(self, args, logger, model_dir, test_dir, setting):
        super(Exp_TimeCAP, self).__init__(args)
        self.logger = logger
        self.model_dir = model_dir
        self.test_dir = test_dir
        self.setting = setting
        # Checkpoints path
        self.model_path = os.path.join(self.model_dir, setting)
        check_dir(self.model_path)
        # Test results path
        self.test_path = os.path.join(self.test_dir, setting)
        check_dir(self.test_path)
        # Best checkpoints path
        self.best_checkpoints_path = os.path.join(self.model_path, f"ckpt_best.pth")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        print("number of model params", sum(p.numel() for p in model.parameters() if p.requires_grad))
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, flag):
        if flag == 'stable_spam':
            model_optim = StableSPAM(self.model.parameters(), lr=self.args.learning_rate, gamma1=0.7, gamma2=0.9, gamma3=0.999, total_T=1000, update_proj_gap=50)
        elif flag == 'adam':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=0)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


    def pretrain(self):

        train_data, train_loaders = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer(self.args.optimizer)
        criterion = self._select_criterion()

        # training
        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
            epoch_time = time.time()

            # Training for Batch Data
            for train_loader in train_loaders:
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(train_loader, file=sys.stdout)):
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = torch.cat([batch_x[:, self.args.pretrain_pred_len:, f_dim:], batch_y[:, self.args.label_len:self.args.label_len + self.args.pretrain_pred_len, f_dim:].to(self.device)], dim=1)

                    # Forecasting
                    outputs = self.model(batch_x, activate_os_head=False)[0]
                    outputs = outputs[:, :, f_dim:]

                    # loss function, 反向传播， 参数更新
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                    loss.backward()
                    model_optim.step()

            # Vali Loss
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print("Epoch: {0}  Spend: {1:.0f} s | Train Loss: {2:.7f}  Vali Loss: {3:.7f}".format(epoch + 1, time.time() - epoch_time, train_loss, vali_loss))
            self.logger.info("Epoch: {0}  Spend: {1:.0f} s | Train Loss: {2:.7f}  Vali Loss: {3:.7f}".format(epoch + 1, time.time() - epoch_time, train_loss, vali_loss))

            # early_stopping
            early_stopping(val_loss = vali_loss, test_loss=None, model = self.model, path = self.best_checkpoints_path, model_name = self.args.model, epoch=epoch + 1, task_name=self.args.task_name, logger = self.logger)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Adjust learning rate
            adjust_learning_rate(optimizer=model_optim, epoch=epoch + 1, args=self.args)

        return early_stopping.val_loss_min


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        ar_total_loss = []
        os_total_loss = []
        self.model.eval()

        if isinstance(vali_loader, list):
            loaders = vali_loader
        else:
            loaders = [vali_loader]

        with torch.no_grad():
            for sub_vali_loader in loaders:
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(sub_vali_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y_AR = torch.cat([batch_x[:, self.args.pretrain_pred_len:, f_dim:], batch_y[:, self.args.label_len:self.args.label_len + self.args.pretrain_pred_len, f_dim:]], dim=1)

                    if self.args.task_name  == 'pretrain':
                        outputs_AR, _, _ = self.model(batch_x, activate_os_head=False)
                        outputs_AR = outputs_AR[:, :, f_dim:]
                        loss = criterion(outputs_AR, batch_y_AR)

                    elif self.args.task_name == 'finetune':
                        batch_y_OS = batch_y[:, self.args.label_len:, f_dim:]
                        outputs_AR, outputs_OS, _ = self.model(batch_x, activate_os_head=True)
                        outputs_AR = outputs_AR[:, :, f_dim:]
                        outputs_OS = outputs_OS[:, :, f_dim:]
                        loss_AR = criterion(outputs_AR, batch_y_AR)
                        loss_OS = criterion(outputs_OS, batch_y_OS)
                        loss_SD = criterion(outputs_AR[:, -self.args.pretrain_pred_len:, :], outputs_OS[:, :self.args.pretrain_pred_len, :])
                        # Sigmoid曲线融合
                        if self.args.use_ar_head and self.args.use_os_head:
                            loss = self.args.lambda1 * loss_AR + self.args.lambda2 * loss_OS + self.args.lambda3 * loss_SD
                        elif self.args.use_ar_head and not self.args.use_os_head:
                            loss = loss_AR
                        elif not self.args.use_ar_head and self.args.use_os_head:
                            loss = loss_OS

                    total_loss.append(loss.item())
                    if self.args.task_name  == 'pretrain':
                        ar_total_loss.append(loss.item())
                    else:
                        ar_total_loss.append(loss_AR.item())
                        os_total_loss.append(loss_OS.item())

        self.model.train()

        print(f"Autoregressive loss: {np.average(ar_total_loss):.4f}")
        print(f"One-shot loss: {np.average(os_total_loss):.4f}")

        return np.average(total_loss)


    def finetune(self):
        # Load Pretraining Models
        if self.args.load_checkpoints:
            print(f'loading from {self.args.best_pretrain_path}')
            state_dict = torch.load(self.args.best_pretrain_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False) # 非严格匹配，忽略state_dict没有但是model有的层, 预训练保存的模型没有OSG_Head的
            param_train = 0
            param_all = 0
            for name, param in self.model.named_parameters(): # 冻住两个attention
                if any(k in name for k in ['attention']):
                    print(name)
                    param.requires_grad = False
                    param_all += param.numel()
                else:
                    param_train += param.numel()
                    param_all += param.numel()
            print(f'trainable parameters num: {clever_format(param_train)}, all parameters num: {clever_format(param_all)},'f'ratio: {param_train / param_all * 100} %')

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=0)
        model_optim = self._select_optimizer(self.args.optimizer)
        criterion = self._select_criterion()

        # training
        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(train_loader, file=sys.stdout)):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y_AR = torch.cat([batch_x[:, self.args.pretrain_pred_len:, f_dim:], batch_y[:, self.args.label_len:self.args.label_len + self.args.pretrain_pred_len, f_dim:].to(self.device)], dim=1)
                batch_y_OS = batch_y[:, self.args.label_len:, f_dim:]

                # Forecasting
                outputs_AR, outputs_OS, _ = self.model(batch_x, activate_os_head=True)
                outputs_AR = outputs_AR[:, :, f_dim:]
                outputs_OS = outputs_OS[:, :, f_dim:]

                # loss function
                loss_AR = criterion(outputs_AR, batch_y_AR)
                loss_OS = criterion(outputs_OS, batch_y_OS)
                loss_SD = criterion(outputs_AR[:, -self.args.pretrain_pred_len:, :].detach(), outputs_OS[:, :self.args.pretrain_pred_len, :]) # 将自回归输出脱离计算图，那么损失就不会传递到AR头上,目的是让一次性生成的前半部分更像AR生成的，实现自蒸馏联合训练
                loss = self.args.lambda1 * loss_AR + self.args.lambda2 * loss_OS + self.args.lambda3 * loss_SD

                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

            # Vali & Test Loss
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print("Epoch: {0}  Spend: {1:.0f} s | Train Loss: {2:.7f}  Vali Loss: {3:.7f}  Test Loss: {4:.7f}".format(epoch + 1, time.time() - epoch_time, train_loss, vali_loss, test_loss))
            self.logger.info("Epoch: {0}  Spend: {1:.0f} s | Train Loss: {2:.7f}  Vali Loss: {3:.7f}  Test Loss: {4:.7f}".format(epoch + 1, time.time() - epoch_time, train_loss, vali_loss, test_loss))

            # Early Stopping
            early_stopping(val_loss = vali_loss, test_loss=test_loss, model = self.model, path = self.best_checkpoints_path, model_name = self.args.model, epoch=epoch + 1, task_name=self.args.task_name, logger = self.logger)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Adjust Learning Rate
            adjust_learning_rate(optimizer=model_optim, epoch=epoch+1, args=self.args)

        return early_stopping.val_loss_min


    def Inference(self):
        # Load test data and model checkpoints
        test_data, test_loader = self._get_data(flag='inference')
        # print("Loading best pretrain model from {}".format(self.best_checkpoints_path))
        # self.model.load_state_dict(torch.load(self.best_checkpoints_path))

        print("Loading best pretrain model from {}".format(self.args.best_pretrain_path))
        self.model.load_state_dict(torch.load(self.args.best_pretrain_path), strict=False)

        # Calculate number of iterations needed for autoregressive prediction
        itrs = int(np.ceil(self.args.pred_len / self.args.pretrain_pred_len))
        preds, trues = [], []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
                pred_tmp = []
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                true = batch_y.detach().cpu().numpy()

                # AR+OS prediction
                for itr in range(itrs):
                    if itr == 0:
                        current_x = batch_x
                        if self.args.use_os_head:
                            outputs_AR, outputs_OS, _ = self.model(current_x, activate_os_head=True)
                        else:
                            outputs_AR, _, _ = self.model(current_x, activate_os_head=False)
                    else:
                        current_x = torch.cat([current_x, pred_tmp[-1]], dim=1)
                        current_x = current_x[:, -self.args.seq_len:, :]
                        outputs_AR, _, _ = self.model(current_x, activate_os_head=False)
                    pred_step = outputs_AR[:, -self.args.pretrain_pred_len:, :]
                    pred_tmp.append(pred_step)

                # Concatenate and truncate predictions to required length
                pred = torch.cat(pred_tmp, dim=1).detach().cpu().numpy()
                pred = pred[:, :self.args.pred_len, f_dim:]

                if self.args.use_ar_head and self.args.use_os_head:
                    point_x = np.linspace(0, 1, self.args.pred_len).reshape(1, self.args.pred_len, 1)
                    weight_os = 1 / (1 + np.exp(-self.args.alpha * (point_x - self.args.beta)))
                    pred_fusion = (1 - weight_os) * pred + weight_os * outputs_OS[:, :, f_dim:].detach().cpu().numpy()
                elif self.args.use_ar_head and not self.args.use_os_head:
                    pred_fusion = pred
                elif not self.args.use_ar_head and self.args.use_os_head:
                    pred_fusion = outputs_OS.detach().cpu().numpy()

                preds.append(pred_fusion)
                trues.append(true)

        # Concatenate all batches
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('Test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # Evaluate prediction
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f'{self.args.seq_len}-pred-{self.args.pred_len}, MSE: {mse:.4f}, MAE: {mae:.4f}')
        self.logger.info(f'{self.args.seq_len}-pred-{self.args.pred_len}, MSE: {mse:.4f}, MAE: {mae:.4f}')

        # Save prediction and metrics
        np.save(os.path.join(self.test_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(self.test_path, 'pred.npy'), preds)
        np.save(os.path.join(self.test_path, 'true.npy'), trues)

        return mse, mae
