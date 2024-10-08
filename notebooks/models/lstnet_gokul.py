"""
LSTNET Model combined from
https://github.com/gokulkarthik/LSTNet.pytorch/blob/master/LSTNet.py
https://github.com/Goochaozheng/LSTNet-Attn/blob/main/models/LSTNet.py#L6
"""
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LSTNet(nn.Module):
    def __init__(self, ar_window_size, num_features, recc1_out_channels, conv1_out_channels):
        super(LSTNet, self).__init__()
        self.ar_window_size = ar_window_size # self.P, tau, window. (5)
        self.num_features = num_features # self.m (7)
        self.recc1_out_channels = recc1_out_channels # self.hidR, number of RNN hidden units (64)
        self.conv1_out_channels = conv1_out_channels # self.hidC, number of CNN hidden units, number of filter (32)
        self.skip_reccs_out_channels = [4, 4] # self.hidS, number of skip-RNN hidden units
        self.conv1_kernel_height = 7 # self.Ck, kernel size of CNN
        self.skip_steps = [4, 24] # self.skip, number of cell that skip through
        # self.pt = int((self.P - self.Ck)/self.skip)     # times of skips
        # self.hw = args.highway_window
        self.output_out_features = 1 # self.hw(?)

        # kernel width is number of variable, covering all variable
        # height is along time axis
        # kernel moving along time axis
        self.conv1 = nn.Conv2d(1, self.conv1_out_channels,
                               kernel_size=(self.conv1_kernel_height, self.num_features))
        # input size equals to output channels of conv1
        self.recc1 = nn.GRU(self.conv1_out_channels, self.recc1_out_channels, batch_first=True)
        self.dropout = nn.Dropout(p = 0.2)

        self.skip_reccs = {}
        for i in range(len(self.skip_steps)):
            self.skip_reccs[i] = nn.GRU(
                self.conv1_out_channels, 
                self.skip_reccs_out_channels[i],
                batch_first=True
                )
        # linear after skip
        # RNN output _ skip-RNN output
        self.output_in_features = (
            self.recc1_out_channels 
            + np.dot(self.skip_steps, self.skip_reccs_out_channels)
            )
        self.output = nn.Linear(self.output_in_features, self.output_out_features)
        if self.ar_window_size > 0:
            self.ar = nn.Linear(self.ar_window_size, 1)

    def forward(self, X):
        """
        Parameters:
        X (tensor) [batch_size, time_steps, num_features]
        """
        batch_size = X.size(0)

        # Convolutional Layer
        C = X.unsqueeze(1) # [batch_size, num_channels=1, time_steps, num_features]
        C = F.relu(self.conv1(C)) # [batch_size, conv1_out_channels, shrinked_time_steps, 1]
        C = self.dropout(C)
        # H = P - Ck + 1, stride = 1
        # output size of conv1 is [batch, hidC, H, 1] for kernel width = input width
        # squeeze the 3-rd dim
        C = torch.squeeze(C, 3) # [batch_size, conv1_out_channels, shrinked_time_steps]

        # Recurrent Layer
        R = C.permute(0, 2, 1) # [batch_size, shrinked_time_steps, conv1_out_channels]
        out, hidden = self.recc1(R) # [batch_size, shrinked_time_steps, recc_out_channels]
        R = out[:, -1, :] # [batch_size, recc_out_channels]
        R = self.dropout(R)
        #print(R.shape)

        # Skip Recurrent Layers
        shrinked_time_steps = C.size(2)
        for i, step in enumerate(self.skip_steps):
            skip_step = step
            skip_sequence_len = shrinked_time_steps // skip_step # self.pt
            # shrinked_time_steps shrinked further
            S = C[:, :, -skip_sequence_len*skip_step:] # [batch_size, conv1_out_channels, shrinked_time_steps]
            S = S.view(S.size(0), S.size(1), skip_sequence_len, skip_step) # [batch_size, conv1_out_channels, skip_sequence_len, skip_step=num_skip_components]
            # note that num_skip_components = skip_step
            S = S.permute(0, 3, 2, 1).contiguous() # [batch_size, skip_step=num_skip_components, skip_sequence_len, conv1_out_channels]
            S = S.view(S.size(0)*S.size(1), S.size(2), S.size(3))  # [batch_size*num_skip_components, skip_sequence_len, conv1_out_channels]
            out, hidden = self.skip_reccs[i](S) # [batch_size*num_skip_components, skip_sequence_len, skip_reccs_out_channels[i]]
            S = out[:, -1, :] # [batch_size*num_skip_components, skip_reccs_out_channels[i]]
            S = S.view(batch_size, skip_step*S.size(1)) # [batch_size, num_skip_components*skip_reccs_out_channels[i]]
            S = self.dropout(S)
            R = torch.cat((R, S), 1) # [batch_size, recc_out_channels + skip_reccs_out_channels * num_skip_components]

        # Output Layer
        O = F.relu(self.output(R)) # [batch_size, output_out_features=1]

        if self.ar_window_size > 0:
            # set dim3 based on output_out_features
            AR = X[:, -self.ar_window_size:, :] # [batch_size, ar_window_size, output_out_features=1] (original=[:, -self.ar_window_size:, 3:4])
            AR = AR.permute(0, 2, 1).contiguous() # [batch_size, output_out_features, ar_window_size]
            AR = self.ar(AR) # [batch_size, output_out_features, 1]
            AR = AR.squeeze(2) # [batch_size, output_out_features]
            O = O + AR

        return O

class LSTNet_lightning(pl.LightningModule):
    def __init__(self, ar_window_size, num_features, recc1_out_channels, conv1_out_channels):
        super(LSTNet_lightning, self).__init__()
        self.ar_window_size = ar_window_size
        self.num_features = num_features
        self.recc1_out_channels = recc1_out_channels
        self.conv1_out_channels = conv1_out_channels
        self.skip_reccs_out_channels = [4, 4]
        self.conv1_kernel_height = 7
        self.skip_steps = [4, 24]
        self.output_out_features = 1

        self.conv1 = nn.Conv2d(1, self.conv1_out_channels,
                               kernel_size=(self.conv1_kernel_height, self.num_features))
        self.recc1 = nn.GRU(self.conv1_out_channels, self.recc1_out_channels, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)

        self.skip_reccs = nn.ModuleList([
            nn.GRU(self.conv1_out_channels, skip_out_channels, batch_first=True)
            for skip_out_channels in self.skip_reccs_out_channels
        ])

        self.output_in_features = self.recc1_out_channels + np.dot(self.skip_steps, self.skip_reccs_out_channels)
        self.output = nn.Linear(self.output_in_features, self.output_out_features)

        if self.ar_window_size > 0:
            self.ar = nn.Linear(self.ar_window_size, 1)

    def forward(self, X):
        batch_size = X.size(0)

        C = X.unsqueeze(1)
        C = F.relu(self.conv1(C))
        C = self.dropout(C)
        C = torch.squeeze(C, 3)

        R = C.permute(0, 2, 1)
        out, _ = self.recc1(R)
        R = out[:, -1, :]
        R = self.dropout(R)

        shrinked_time_steps = C.size(2)
        for i, step in enumerate(self.skip_steps):
            skip_step = step
            skip_sequence_len = shrinked_time_steps // skip_step
            S = C[:, :, -skip_sequence_len * skip_step:]
            S = S.view(S.size(0), S.size(1), skip_sequence_len, skip_step)
            S = S.permute(0, 3, 2, 1).contiguous()
            S = S.view(batch_size * S.size(1), S.size(2), S.size(3))
            out, _ = self.skip_reccs[i](S)
            S = out[:, -1, :]
            S = S.view(batch_size, skip_step * S.size(1))
            S = self.dropout(S)
            R = torch.cat((R, S), 1)

        O = F.relu(self.output(R))

        if self.ar_window_size > 0:
            AR = X[:, -self.ar_window_size:, :]
            AR = AR.permute(0, 2, 1).contiguous()
            AR = self.ar(AR)
            AR = AR.squeeze(2)
            O = O + AR

        return O

    def training_step(self, batch, batch_idx):
        X, y = batch
        outputs = self(X)
        loss = F.mse_loss(outputs, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self, lr=1e-3, weight_decay=1e-2):
        return optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
