import torch
import torch.nn as nn
from layers.TimeCAP_EncDec import DataEmbedding_MultiScale
from layers.TimeCAP_EncDec import ChannelLayer, ChannelBlock, AttentionLayer, ChannelAttention
from layers.TimeCAP_EncDec import Decoder_TimeCAP

# Multi-Scale Learning Block
class MultiscaleBlock(nn.Module):
    def __init__(self, configs, depth):
        super().__init__()
        self.configs = configs
        self.window_size = configs.window_size[depth]
        self.stride_channel = configs.stride_channel[depth]

        self.patch_len = configs.patch_len[depth]
        self.stride_time = configs.stride_time[depth]
        assert configs.seq_len % self.patch_len == 0, f"RCR: Dimension mismatch (seq_len & patch_len)"
        self.patch_num = int((configs.seq_len - self.patch_len) / self.stride_time) + 1

        # Multi-Scale Embedding
        self.embedding = DataEmbedding_MultiScale(window_size=self.window_size, stride_channel=self.stride_channel, patch_len=self.patch_len, stride_time=self.stride_time, patch_num=self.patch_num, d_model=configs.d_model, dropout=configs.dropout)

        # Encoder
        self.encoder = ChannelBlock(
            [
                ChannelLayer(
                    attention=AttentionLayer(
                        ChannelAttention(mask_flag=True, attention_dropout=configs.dropout, output_attention=configs.output_attention, d_model=configs.d_model, num_heads=configs.n_heads, scope=configs.scope, covariate=configs.covariate, flash_attention=configs.flash_attention),
                        configs.d_model,
                        configs.n_heads),
                    cross_attention=AttentionLayer(
                        ChannelAttention(mask_flag=True, attention_dropout=configs.dropout, output_attention=configs.output_attention, d_model=configs.d_model, num_heads=configs.n_heads, scope=configs.scope, covariate=configs.covariate, flash_attention=configs.flash_attention),
                        configs.d_model,
                        configs.n_heads),
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=None,
        )

        # Decoder
        has_os_head = True if configs.task_name == 'finetune' and depth == configs.depth - 1 else False
        self.decoder = Decoder_TimeCAP(d_model=configs.d_model, patch_len=self.patch_len, patch_num=self.patch_num, stride_channel=self.stride_channel, OS_pred_lenth=configs.pred_len, has_os_head=has_os_head, dropout=configs.dropout)

    def forward(self, x, activate_os_head): # b, l, c
        B, L, C = x.shape
        assert (C - self.window_size) % self.stride_channel == 0, f"RCR: Dimension mismatch (window_size & stride_channel)"
        group_num = int((C - self.window_size) / self.stride_channel) + 1

        # Multi-Scale Embedding
        x_embedding = self.embedding(x) # bk, (w+1)p, d

        # Encoder
        enc_out, attns = self.encoder(x_embedding, n_vars=self.window_size + 1, n_tokens=self.patch_num, group_num=group_num) # bk, (w+1)p, d

        # Decoder
        dec_out_AR, dec_out_OS = self.decoder(enc_out, group_num=group_num, enc_in=C, activate_os_head=activate_os_head) # b, c, l | b, c, h (None)

        return dec_out_AR.permute(0, 2, 1), dec_out_OS, attns # b, l, c


# Our Model for AAAI
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.learners = nn.ModuleList([MultiscaleBlock(configs, i) for i in range(configs.depth)])

    def forecast(self, batch_x, activate_os_head=False):
        # Instance Normalization
        means = torch.mean(batch_x, dim=1, keepdim=True)
        batch_x = batch_x - means
        stdev = torch.sqrt(torch.var(batch_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        batch_x = batch_x / stdev  # b, l, c

        # Multi-Scale
        attns, output_OS = (None, None)
        for i, learner in enumerate(self.learners):
            batch_x, output_OS, attns = learner(batch_x, activate_os_head)

        # Instance De-Normalization
        dec_out_AR = batch_x
        dec_out_AR = dec_out_AR * (stdev[:, 0, :].unsqueeze(1))  # b, 1, c
        dec_out_AR = dec_out_AR + (means[:, 0, :].unsqueeze(1))  # b, 1, c

        dec_out_OS = output_OS
        if activate_os_head:
            dec_out_OS = dec_out_OS * (stdev[:, 0, :].unsqueeze(1))  # b, 1, c
            dec_out_OS = dec_out_OS + (means[:, 0, :].unsqueeze(1))  # b, 1, c

        return dec_out_AR, dec_out_OS, attns

    def forward(self, batch_x, activate_os_head, mask=None):
        if self.configs.task_name == 'pretrain':
            dec_out_AR, dec_out_OS, attns = self.forecast(batch_x, activate_os_head)
            return dec_out_AR, dec_out_OS, attns

        if self.configs.task_name == 'finetune' and self.configs.downstream_task == 'forecasting':
            dec_out_AR, dec_out_OS, attns = self.forecast(batch_x, activate_os_head)
            return dec_out_AR, dec_out_OS, attns

        return None

