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

class NARNN(nn.Module):
    """
    Nonlinear Autoregressive Neural Network (NAR-NN)
    Predicts x[t] from [x[t-1], x[t-2], ..., x[t-p]] using an MLP.

    Input:
      Assumes input x of shape:
        (batch, win_len, 1)
    """

    def __init__(self, win_len, hidden_size, num_layers, device=torch.device("cuda")):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=1)
        # Build MLP with num_layers hidden layers
        layers = []
        layers.append(nn.Linear(in_features=win_len, out_features=hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x, h=None):
        x_norm = self.bn(x)
        x_norm = x_norm.permute(0, 2, 1).contiguous()
        x_flat = x_norm.reshape(x_norm.size(0), -1)
        y = self.mlp(x_flat)
        y = self.fc(y)
        return y


class NARNN_v2(nn.Module):
    def __init__(self, win_len, hidden_size, num_layers):
        super().__init__()
        self.bn = nn.BatchNorm1d(1)

        # Replace flatten with 1D convolutions
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(hidden_size // 4)  # Compress sequence
        )

        self.mlp = nn.Sequential(
            nn.Linear(64 * (hidden_size // 4), hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, h=None):
        x_norm = self.bn(x)
        x_features = self.conv_encoder(x_norm)  # Preserve temporal structure
        x_flat = x_features.flatten(1)
        y = self.mlp(x_flat)
        return self.fc(y)


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

class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell that uses convolution operations in both
    input-to-state and state-to-state transitions.

    Based on: "Convolutional LSTM Network: A Machine Learning Approach
    for Precipitation Nowcasting" (Shi et al., 2015)
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Combined convolution for all gates: input, forget, cell, output
        # This is more efficient than separate convolutions
        self.conv = nn.Conv1d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Xavier initialization for weights
        nn.init.xavier_uniform_(self.conv.weight)
        # Initialize biases
        # Forget gate bias to 1 (helps with gradient flow)
        nn.init.constant_(self.conv.bias, 0)
        nn.init.constant_(self.conv.bias[self.hidden_channels:2*self.hidden_channels], 1.0)

    def forward(self, x, hidden_state):
        """
        Args:
            x: input tensor of shape (batch, input_channels, seq_len)
            hidden_state: tuple of (h, c) where each is (batch, hidden_channels, seq_len)

        Returns:
            h_next: next hidden state (batch, hidden_channels, seq_len)
            c_next: next cell state (batch, hidden_channels, seq_len)
        """
        h_prev, c_prev = hidden_state

        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)  # (batch, input+hidden, seq_len)

        # Apply convolution and split into gates
        gates = self.conv(combined)  # (batch, 4*hidden_channels, seq_len)

        # Split into individual gates
        i_gate, f_gate, g_gate, o_gate = torch.split(gates, self.hidden_channels, dim=1)

        # Apply activations
        i_gate = torch.sigmoid(i_gate)  # Input gate
        f_gate = torch.sigmoid(f_gate)  # Forget gate
        g_gate = torch.tanh(g_gate)     # Cell gate
        o_gate = torch.sigmoid(o_gate)  # Output gate

        # Update cell state
        c_next = f_gate * c_prev + i_gate * g_gate

        # Update hidden state
        h_next = o_gate * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, seq_len, device):
        """Initialize hidden and cell states"""
        h = torch.zeros(batch_size, self.hidden_channels, seq_len, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, seq_len, device=device)
        return (h, c)


class ConvLSTM(nn.Module):
    """
    Multi-layer Convolutional LSTM network based on LSTMO structure.
    Uses convolution operations inside LSTM for both input-to-state
    and state-to-state transitions.
    """
    def __init__(self, input_channels=1, hidden_channels=64, num_layers=2,
                 kernel_size=3, device=torch.device("cuda")):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        # Batch normalization for input
        self.bn = nn.BatchNorm1d(num_features=input_channels)

        # Create ConvLSTM cells for each layer
        self.conv_lstm_cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            self.conv_lstm_cells.append(
                ConvLSTMCell(in_ch, hidden_channels, kernel_size)
            )

        # Global average pooling to reduce sequence dimension
        self.global_pool = FastGlobalAvgPool(flatten=True)

        # Final fully connected layer
        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self, x, h=None):
        """
        Args:
            x: input tensor of shape (batch, input_channels, seq_len)
            h: optional hidden state tuple of lists [(h1, c1), (h2, c2), ...]

        Returns:
            output: prediction of shape (batch, 1)
        """
        batch_size, _, seq_len = x.shape

        # Normalize input
        x_norm = self.bn(x)

        # Initialize hidden states if not provided
        if h is None:
            h = [cell.init_hidden(batch_size, seq_len, x.device)
                 for cell in self.conv_lstm_cells]

        # Process through ConvLSTM layers
        current_input = x_norm
        new_h = []

        for layer_idx, cell in enumerate(self.conv_lstm_cells):
            h_next, c_next = cell(current_input, h[layer_idx])
            new_h.append((h_next, c_next))
            current_input = h_next  # Output of this layer is input to next

        # Global average pooling over sequence dimension
        # current_input shape: (batch, hidden_channels, seq_len)
        pooled = self.global_pool(current_input)  # (batch, hidden_channels)

        # Final prediction
        output = self.fc(pooled)  # (batch, 1)

        return output


class CNNLSTM(nn.Module):
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


class TransformerModelV03(nn.Module):
    def __init__(self, input_dim=1, nf=16, model_dim=128, num_heads=8, num_layers=2, output_dim=1):
        super().__init__()
        drop_rate = 0.1
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, nf, 5, 2, 1, bias=False),
            nn.BatchNorm1d(nf),
            Down(c_in=nf, c_out=nf * 2, ds=2),
            ResBlock(dim=nf * 2),
            Down(c_in=nf * 2, c_out=nf * 4, ds=2),
            ResBlock(dim=nf * 4)
        )
        self.pos_emb = nn.Conv1d(model_dim, model_dim, 1, 1)  # positional embedding
        enc_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, activation="gelu",
                                               dim_feedforward=2 * model_dim, dropout=drop_rate, batch_first=True)
        self.tf = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False, norm=nn.LayerNorm(model_dim))
        self.fc = nn.Linear(model_dim, output_dim)  # Fully connected layer for output
        self.avg_pool = FastGlobalAvgPool(flatten=True)

    def forward(self, x):
        # pass the input through several convolution layers to down sample the data while preserving the local
        # spatial information
        x_conved = self.conv_block(x)  # out Shape: (batch_size, model_dim, seq_length/8)
        # Embed the input
        x_pos_embedded = self.pos_emb(x_conved)  # out Shape: (batch_size, model_dim, seq_length/8)
        # Reshape for Transformer: (batch_size, seq_length/8, model_dim)
        x_transformed = x_pos_embedded.permute(0, 2, 1).contiguous()
        # Pass through the transformer encoder
        y = self.tf(x_transformed)  # out Shape: (batch_size, seq_length/8, model_dim)
        # Reshape for Average Pooling: (batch_size, model_dim, seq_length/8)
        y_perm = y.permute(0, 2, 1).contiguous()
        # Final output layer
        y_avg_pool = self.avg_pool(y_perm)  # out shape (batch_size, model_dim)
        out = self.fc(y_avg_pool)  # Shape: (batch_size, output_dim)
        return out


class TransformerModelV04(nn.Module):
    def __init__(self, input_dim=1, nf=16, model_dim=128, num_heads=8, num_layers=2, output_dim=1):
        super().__init__()
        drop_rate = 0.1
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, nf, 5, 2, 1, bias=False),
            nn.BatchNorm1d(nf),
            Down(c_in=nf, c_out=nf * 2, ds=2),
            ResBlock(dim=nf * 2),
            Down(c_in=nf * 2, c_out=model_dim, ds=2),
            ResBlock(dim=model_dim)
        )
        self.pos_emb = nn.Conv1d(model_dim, model_dim, 1, 1)  # positional embedding
        enc_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, activation="gelu",
                                               dim_feedforward=2 * model_dim, dropout=drop_rate, batch_first=True)
        self.tf = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False, norm=nn.LayerNorm(model_dim))
        self.fc = nn.Linear(model_dim, output_dim)  # Fully connected layer for output
        self.avg_pool = FastGlobalAvgPool(flatten=True)

    def forward(self, x):
        # pass the input through several convolution layers to down sample the data while preserving the local
        # spatial information
        x_conved = self.conv_block(x)  # out Shape: (batch_size, model_dim, seq_length/8)
        # Embed the input
        x_pos_embedded = self.pos_emb(x_conved)  # out Shape: (batch_size, model_dim, seq_length/8)
        # Reshape for Transformer: (batch_size, seq_length/8, model_dim)
        x_transformed = x_pos_embedded.permute(0, 2, 1).contiguous()
        # Pass through the transformer encoder
        y = self.tf(x_transformed)  # out Shape: (batch_size, seq_length/8, model_dim)
        # Reshape for Average Pooling: (batch_size, model_dim, seq_length/8)
        y_perm = y.permute(0, 2, 1).contiguous()
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
