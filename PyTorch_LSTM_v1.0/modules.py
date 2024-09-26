import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.padding import ReflectionPad1d
from torch.nn.utils import weight_norm
import torch.nn.init as init


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(classname, torch.nn.Conv1d) or isinstance(classname, torch.nn.Conv2d):
        nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)
    if classname.find("Linear") != -1:
        nn.init.orthogonal_(m.weight)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


# class FastGlobalAvgPool(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         in_size = x.size()
#         return x.view((in_size[0], in_size[1], -1)).mean(dim=2)


class FastGlobalAvgPool(nn.Module):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1)


class ResBlock(nn.Module):
    def __init__(self, dim, dilation=1, ks=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad1d(ks // 2 * dilation),
            nn.Conv1d(dim, dim, kernel_size=ks, dilation=dilation, bias=False),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(dim, dim, kernel_size=1, dilation=dilation),
        )

    def forward(self, x):
        return x + self.block(x)


class Down(nn.Module):
    def __init__(self, c_in, c_out, ds):
        super().__init__()
        ks = ds + 1
        self.block = nn.Sequential(nn.ReflectionPad1d(ks // 2),
                                   nn.Conv1d(in_channels=c_in,
                                             out_channels=c_out,
                                             kernel_size=ks,
                                             stride=ds,
                                             padding=0,
                                             bias=False),
                                   nn.BatchNorm1d(c_out),
                                   nn.LeakyReLU(0.2, True),
                                   )

    def forward(self, x):
        x = self.block(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ngf = 16
        model = []
        model += [
            nn.ReflectionPad1d(1),
            nn.Conv1d(in_channels=1, out_channels=ngf, kernel_size=3, stride=1, padding=0, groups=1, bias=False),
            nn.BatchNorm1d(ngf),
            nn.LeakyReLU(0.2, True),
        ]
        c_in = ngf
        for _ in range(3):
            c_out = int(c_in)
            model += [
                ResBlock(dim=c_out, dilation=1, ks=3),
                ResBlock(dim=c_out, dilation=3, ks=3),
                ResBlock(dim=c_out, dilation=9, ks=3),
            ]
            c_in = c_out
        model += [nn.Conv1d(in_channels=c_in, out_channels=1, kernel_size=1, stride=1, padding=0, groups=1, bias=False)]
        self.conv_stack = nn.Sequential(*model)

        def _initialize_params(m):
            classname = m.__class__.__name__
            if classname.find('Conv1d') != -1:
                with torch.no_grad():
                    m.weight.data.normal_(1.0, 0.02)
                    if m.bias is not None:
                        m.bias.fill_(0)

            elif classname.find('BatchNorm1d') != -1:
                with torch.no_grad():
                    m.weight.data.normal_(1.0, 0.02)
                    m.bias.fill_(0)
            else:
                pass

        self.apply(_initialize_params)

    def forward(self, x):
        x = self.conv_stack(x)
        return x


class LSTMO(nn.Module):
    def __init__(self, device=torch.device("cuda")):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=1)
        self.l = nn.LSTM(batch_first=True, hidden_size=128, num_layers=2, input_size=1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x, h=None):
        x_norm = self.bn(x)
        x_norm = x_norm.permute(0, 2, 1).contiguous()
        y, h = self.l(x_norm)
        y = self.fc(y[:, -1, :])
        return y


class ConvLSTM(nn.Module):
    def __init__(self, device=torch.device("cuda")):
        super().__init__()
        nf = 16  # Number of filters in CNN
        self.conv_stack = nn.Sequential(
            nn.ReflectionPad1d(1),
            nn.Conv1d(1, nf, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(nf),
            nn.LeakyReLU(0.2, True),
            ResBlock(dim=nf, dilation=1, ks=3),
            ResBlock(dim=nf, dilation=3, ks=3),
            ResBlock(dim=nf, dilation=9, ks=3),
        )
        self.l = nn.LSTM(batch_first=True, hidden_size=128, num_layers=2, input_size=nf)
        self.fc = nn.Linear(128, 1)

    def forward(self, x, h=None):
        x = self.conv_stack(x)
        x = x.permute(0, 2, 1).contiguous()
        y, h = self.l(x)
        y = self.fc(y[:, -1, :])
        return y


class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, model_dim=256, num_heads=8, num_layers=2, output_dim=1,
                 device=torch.device("cuda")):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=input_dim)  # Batch normalization for the input
        self.embedding = nn.Linear(input_dim, model_dim)  # Linear layer for embedding input
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(model_dim, output_dim)  # Fully connected layer for output

    def forward(self, x):
        # Normalize the input
        x_norm = self.bn(x)  # Shape: (batch_size, seq_length, input_dim)
        # Permute to (batch_size, sequence_length, input_dim)
        x_norm = x_norm.permute(0, 2, 1)  # Shape: (batch_size, sequence_length, input_dim)
        # Add an extra dimension if input is 2D (batch_size, seq_length)
        if x_norm.ndim == 2:
            x_norm = x_norm.unsqueeze(-1)  # Shape: (batch_size, seq_length, input_dim)
        # Embed the input
        x_embedded = self.embedding(x_norm)  # Shape: (batch_size, seq_length, model_dim)
        # Reshape for Transformer: (seq_length, batch_size, model_dim)
        x_transformed = x_embedded.permute(1, 0, 2)

        # Pass through the transformer encoder
        y = self.transformer_encoder(x_transformed)

        # Get the output for the last time step
        y = y[-1, :, :]  # Shape: (batch_size, model_dim)

        # Final output layer
        y = self.fc(y)  # Shape: (batch_size, output_dim)

        return y


class TransformerModelV02(nn.Module):
    def __init__(self, input_dim=1, model_dim=64, num_heads=4, num_layers=2, output_dim=1,
                 device=torch.device("cuda")):
        super().__init__()
        # self.num_tokens = 1
        drop_rate = 0.1
        self.bn = nn.BatchNorm1d(num_features=input_dim)  # Batch normalization for the input
        # self.embedding = nn.Linear(input_dim, model_dim)  # Linear layer for embedding input
        # maybe need to replace with several convolution layers
        self.pos_emb = nn.Conv1d(input_dim, model_dim, 1, 1)  # positional embedding

        enc_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, activation="gelu",
                                               dim_feedforward=2 * model_dim, dropout=drop_rate, batch_first=True)
        self.tf = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, 64))
        self.fc = nn.Linear(model_dim, output_dim)  # Fully connected layer for output
        self.avg_pool = FastGlobalAvgPool(flatten=True)

    def forward(self, x):
        # Normalize the input
        x_norm = self.bn(x)  # Shape: (batch_size, input_dim, seq_length)
        # Embed the input
        x_pos_embedded = self.pos_emb(x_norm)  # out Shape: (batch_size, model_dim, seq_length)
        # Reshape for Transformer: (batch_size, seq_length, model_dim)
        x_transformed = x_pos_embedded.permute(0, 2, 1)
        # Pass through the transformer encoder
        y = self.tf(x_transformed)  # out Shape: (batch_size, seq_length, model_dim)
        # Reshape for Average Pooling: (batch_size, model_dim, seq_length)
        y_perm = y.permute(0, 2, 1)
        # Final output layer
        y_avg_pool = self.avg_pool(y_perm)  # out shape (batch_size, model_dim)
        out = self.fc(y_avg_pool)  # Shape: (batch_size, output_dim)
        return out


if __name__ == "__main__":
    b = 2
    x = torch.randn(b, 1, 32)
    net = Net()
    y = net(x)
    print(y.shape)
