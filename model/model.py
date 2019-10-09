import torch.nn as nn
import torch.nn.functional as F
import torch
from base import BaseModel
import numpy as np


class WaveNetModel(BaseModel):
    def __init__(self,
                 n_layers=10,
                 n_blocks=4,
                 n_dilation_channels=32,
                 n_residual_channels=32,
                 n_skip_channels=256,
                 n_end_channels=256,
                 n_classes=256,
                 output_length=32,
                 kernel_size=2,
                 bias=False):
        super().__init__()

        # Parameters
        self.n_layers = n_layers
        self.n_blocks = n_blocks
        self.n_dilation_channels = n_dilation_channels
        self.n_residual_channels = n_residual_channels
        self.n_skip_channels = n_skip_channels
        self.n_end_channels = n_end_channels
        self.n_classes = n_classes
        self.kernel_size = kernel_size

        # build model
        self.receptive_field = self.calc_receptive_fields(self.n_layers, self.n_blocks)

        # 1x1 convolution to create channels
        self.causal = CausalConv1d(self.n_classes, self.n_residual_channels, kernel_size=kernel_size, bias=bias)

        # Residual block
        self.res_blocks = ResidualStack(self.n_layers,
                                        self.n_blocks,
                                        self.n_residual_channels,
                                        self.n_classes,
                                        self.kernel_size,
                                        bias)

        self.end_net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(self.n_classes, self.n_end_channels, kernel_size=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.n_end_channels, self.n_end_channels, kernel_size=1, bias=bias)
        )

        self.output_length = output_length

    def forward(self, input):
        x = self.causal(input)
        skip_connections = self.res_blocks(x, self.output_length)
        output = torch.sum(skip_connections, dim=0)
        output = self.end_net(output)

        return output

    @staticmethod
    def calc_receptive_fields(layer_size, stack_size):
        # TODO: check
        layers = [2 ** i for i in range(0, layer_size)] * stack_size
        num_receptive_fields = np.sum(layers)

        return int(num_receptive_fields)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, bias=False):
        super(CausalConv1d, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=1, padding=1,
                              bias=bias)

    def forward(self, x):
        output = self.conv(x)

        # remove last value for causal convolution! (It should not use the last value to not mix train and test)
        return output[:, :, :-self.conv.padding[0]]


class ResidualStack(torch.nn.Module):
    def __init__(self, layer_size, stack_size, res_channels, skip_channels, kernel_size=2, bias=False):
        """
        Stack residual blocks by layer and stack size
        :param layer_size: integer, 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
        :param stack_size: integer, 5 = stack[layer1, layer2, layer3, layer4, layer5]
        :param res_channels: number of residual channel for input, output
        :param skip_channels: number of skip channel for output
        :return:
        """
        super(ResidualStack, self).__init__()

        self.layer_size = layer_size
        self.stack_size = stack_size

        self.res_blocks = self.stack_res_block(res_channels, skip_channels, kernel_size, bias)

    def build_dilations(self):
        dilations = []

        # 5 = stack[layer1, layer2, layer3, layer4, layer5]
        for s in range(0, self.stack_size):
            # 10 = layer[dilation=1, dilation=2, 4, 8, 16, 32, 64, 128, 256, 512]
            for l in range(0, self.layer_size):
                dilations.append(2 ** l)

        return dilations

    def stack_res_block(self, res_channels, skip_channels, kernel_size, bias):
        """
        Prepare dilated convolution blocks by layer and stack size
        """
        res_blocks = []
        dilations = self.build_dilations()

        for d in dilations:
            res_blocks.append(ResidualBlock(res_channels,
                                            skip_channels,
                                            d,
                                            kernel_size=kernel_size,
                                            bias=bias))

        return res_blocks

    def forward(self, x, skip_size):
        """
        :param x: Input for the operation
        :param skip_size: The last output size for loss and prediction
        :return:
        """
        output = x
        skip_connections = []

        for res_block in self.res_blocks:
            # output is the next input
            output, skip = res_block(output, skip_size)
            skip_connections.append(skip)

        return torch.stack(skip_connections)


class ResidualBlock(torch.nn.Module):

    def __init__(self, res_channels: int, skip_channels, dilation, kernel_size, bias=False):
        """
        Thanks to https://github.com/golbin/WaveNet

        :param res_channels: number of residual channels
        :param skip_channels: number of skip channels
        :param dilation: dilation size
        :param kernel_size: kernel size
        :param bias: is there the bias?
        """
        super(ResidualBlock, self).__init__()

        self.dilated = DilatedCausalConv1d(res_channels, dilation=dilation, kernel_size=kernel_size, bias=bias)
        self.conv_res = torch.nn.Conv1d(res_channels, res_channels, 1, bias=bias)
        self.conv_skip = torch.nn.Conv1d(res_channels, skip_channels, 1, bias=bias)

        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()

    def forward(self, x, skip_size):
        """
        :param x: Input of the residual block
        :param skip_size: The last output size for loss and prediction
        :return:
        """
        output = self.dilated(x)

        # PixelCNN gate
        gated_tanh = self.gate_tanh(output)
        gated_sigmoid = self.gate_sigmoid(output)
        gated = gated_tanh * gated_sigmoid

        # Residual network
        output = self.conv_res(gated)
        input_cut = x[:, :, -output.size(2):]
        output += input_cut

        # Skip connection
        skip = self.conv_skip(gated)
        skip = skip[:, :, -skip_size:]

        return output, skip


class DilatedCausalConv1d(torch.nn.Module):
    """Dilated Causal Convolution for WaveNet """
    def __init__(self, channels, kernel_size=2, dilation=1, bias=False):
        """
        Thanks to https://github.com/golbin/WaveNet

        :param channels: number of channels for the CausalConv
        :param kernel_size: kernel size
        :param dilation: dilation size
        :param bias: Is there a bias?
        """
        super(DilatedCausalConv1d, self).__init__()

        pad = (kernel_size - 1) * dilation

        self.conv = torch.nn.Conv1d(channels, channels,
                                    kernel_size=kernel_size,
                                    stride=1,  # Fixed for WaveNet
                                    dilation=dilation,
                                    padding=pad,
                                    bias=bias)

    def forward(self, x):
        output = self.conv(x)

        return output[:, :, :-self.conv.padding[0]]
