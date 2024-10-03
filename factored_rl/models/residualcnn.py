import torch
import torchvision
from torch import nn
import numpy as np
import math

from factored_rl.models.nnutils import Module

##### Doubling Depth Residual ConvNet

ACTIVATION_LAYERS = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
    "mish": nn.Mish,
}


def get_activation_layer(activation: str):
    if activation in ACTIVATION_LAYERS:
        return ACTIVATION_LAYERS[activation]
    else:
        raise ValueError(f"Unknown activation layer: {activation}")


class _ResidualBlock(Module):
    def __init__(
        self,
        input_shape,
        out_channels,
        cnn_blocks=2,
        activation="silu",
        transposed=False,
        k=4,
        s=2,
    ):
        super().__init__()
        self.input_shape = input_shape
        in_channels, width, height = input_shape
        layers = []
        layer_norm_fn = nn.LayerNorm
        layer_norm_fn = nn.RMSNorm
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s)
            if not transposed
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k, stride=s),
            layer_norm_fn(
                [out_channels, int((width - k) / s + 1), int((height - k) / s + 1)]
            )
            if not transposed
            else nn.LayerNorm(
                [out_channels, (width - 1) * s + k, (height - 1) * s + k]
            ),
            get_activation_layer(activation)(),
        )
        for i in range(cnn_blocks):
            layers.append(
                layer_norm_fn(
                    [out_channels, int((width - k) / s + 1), int((height - k) / s + 1)]
                )
                if not transposed
                else nn.LayerNorm(
                    [out_channels, (width - 1) * s + k, (height - 1) * s + k]
                )
            )
            layers.append(get_activation_layer(activation)())
            layers.append(
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=1, stride=1, padding="same"
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.pre_conv(x)
        _x = x
        for layer in self.layers:
            _x = layer(_x)
        return x + _x


class ResidualEncoderNet(Module):
    def __init__(
        self,
        input_shape=(1, 40, 40),
        cnn_blocks=2,
        depth=24,
        min_resolution=4,
        cnn_activation="silu",
        mlp_layers=[],
        mlp_activation="silu",
    ):
        super().__init__()
        self.input_shape = input_shape
        self.cnn_blocks = cnn_blocks
        self.depth = depth
        self.min_resolution = min_resolution
        self.activation = cnn_activation

        # build network.

        n_res_layers = int(np.log2(self.input_shape[-1]) - np.log2(min_resolution))
        print(f"Building Residual Encoder with {n_res_layers} layers.")
        layers = []
        k = 3
        s = 2
        for i in range(n_res_layers):
            layers.append(
                _ResidualBlock(input_shape, depth, cnn_blocks, cnn_activation, k=k, s=s)
            )
            input_shape = (
                depth,
                int((input_shape[1] - k) / s + 1),
                int((input_shape[1] - k) / s + 1),
            )
            depth *= 2

        self.final_shape = input_shape

        self.layers = nn.ModuleList(layers)

        # ---- MLP start
        self.out_dim = input_shape[0] * input_shape[1] * input_shape[2]
        _outdim = mlp_layers[-1]
        mlp_layers = [self.out_dim] + mlp_layers[:-1]
        mlp_layers = list(zip(mlp_layers[:-1], mlp_layers[1:]))
        layers = []
        self.outdim = self.out_dim

        layer_norm_fn = nn.LayerNorm
        layer_norm_fn = nn.RMSNorm
        for in_dim, out_dim in mlp_layers:
            layers.append(layer_norm_fn(in_dim))
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(get_activation_layer(mlp_activation)())

        layers.append(nn.Linear(mlp_layers[-1][-1], _outdim))
        self.outdim = _outdim
        self.mlp = nn.Sequential(*layers)

    def conditioning(self, x, z):
        batch_dims = z.shape[:-1]
        _z = z.reshape(-1, z, 1, 1).repeat(1, 1, *self.input_shape[1:])
        return torch.cat([x, _z], dim=1)

    def forward(self, x, cond=None):
        batch_dims, input_shape = x.shape[:-3], x.shape[-3:]
        x = x.reshape(-1, *input_shape)
        if cond is not None:
            x = self.conditioning(x, cond)
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(x.shape[0], -1)
        x = self.mlp(x)
        return x.reshape(*batch_dims, -1)


class ResidualDecoderNet(Module):
    def __init__(
        self,
        output_shape=(1, 40, 40),
        cnn_blocks=2,
        depth=24,
        min_resolution=4,
        cnn_activation="silu",
        mlp_layers=[],
        mlp_activation="silu",
    ):
        super().__init__()
        self.input_shape = output_shape
        self.cnn_blocks = cnn_blocks
        self.depth = depth
        self.min_resolution = min_resolution
        self.activation = cnn_activation

        # build network.
        n_res_layers = int(np.log2(self.input_shape[-1]) - np.log2(self.min_resolution))
        print(f"Building Residual Decoder with {n_res_layers} layers.")
        layers = []
        depth = int(depth * 2 ** (n_res_layers - 1))
        min_size = self.input_shape[-1]
        for i in range(n_res_layers):
            min_size = (min_size - 4) / 2 + 1
        print(min_size)
        min_size = math.ceil(min_size)
        self.initial_shape = (depth, min_size, min_size)
        assert depth == self.initial_shape[0]
        depth = depth // 2
        output_shape = self.initial_shape

        for i in range(n_res_layers):
            layers.append(
                _ResidualBlock(
                    output_shape, depth, cnn_blocks, cnn_activation, transposed=True
                )
            )
            # output_shape = (depth, int((output_shape[1]-4) / 2 + 1),int((output_shape[1]-4) / 2 + 1))
            output_shape = (
                depth,
                (output_shape[1] - 1) * 2 + 4,
                (output_shape[1] - 1) * 2 + 4,
            )
            depth = depth // 2 if i != n_res_layers - 2 else self.input_shape[0]

        self.layers = nn.ModuleList(layers)

        mlp_layers = mlp_layers + [
            self.initial_shape[0] * self.initial_shape[1] * self.initial_shape[2]
        ]
        mlp_layers = list(zip(mlp_layers[:-1], mlp_layers[1:]))
        layers = []
        for in_dim, out_dim in mlp_layers:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(get_activation_layer(mlp_activation)())
            self.outdim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.crop = torchvision.transforms.CenterCrop(self.input_shape[1:])

    def forward(self, x):
        batch_dims, input_shape = x.shape[:-1], x.shape[-1:]
        x = x.reshape(-1, *input_shape)
        x = self.mlp(x)  # up projection
        x = x.reshape(-1, *self.initial_shape)
        for layer in self.layers:
            x = layer(x)
        x = self.crop(x)
        return x.reshape(*batch_dims, *x.shape[-3:])
