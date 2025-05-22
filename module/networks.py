import collections
from collections.abc import Sequence

from typing import Literal

import torch

from torch import nn
from torch.nn.functional import interpolate



class Permute(nn.Module):
    """Permute the dimensions of the input tensor.
    For example:
    ```
    Permute('NTC', 'NCT') == x.permute(0, 2, 1)
    ```
    """

    def __init__(self, from_dims: str, to_dims: str) -> None:
        super().__init__()
        assert len(from_dims) == len(
            to_dims
        ), "Same number of from- and to- dimensions should be specified for"

        if len(from_dims) not in {3, 4, 5, 6}:
            raise ValueError(
                "Only 3, 4, 5, and 6D tensors supported in Permute for now"
            )

        self.from_dims = from_dims
        self.to_dims = to_dims
        self._permute_idx: list[int] = [from_dims.index(d) for d in to_dims]

    def get_inverse_permute(self) -> "Permute":
        "Get the permute operation to get us back to the original dim order"
        return Permute(from_dims=self.to_dims, to_dims=self.from_dims)

    def __repr__(self):
        return f"Permute({self.from_dims!r} => {self.to_dims!r})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self._permute_idx)


class BatchNorm1d(nn.Module):
    """Wrapper around nn.BatchNorm1d except in NTC format"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.permute_forward = Permute("NTC", "NCT")
        self.bn = nn.BatchNorm1d(*args, **kwargs)
        self.permute_back = Permute("NCT", "NTC")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.permute_back(self.bn(self.permute_forward(inputs)))


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        norm_type: Literal["layer", "batch", "none"] = "layer",
        dropout: float = 0.0,
    ):
        """A 1D convolution with padding so the input and output lengths match."""

        super().__init__()

        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        layers = {}
        layers["conv1d"] = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        if norm_type == "batch":
            layers["norm"] = BatchNorm1d(out_channels)

        layers["relu"] = nn.ReLU(inplace=True)
        layers["dropout"] = nn.Dropout(dropout)

        self.conv = nn.Sequential(
            *[layers[key] for key in layers if layers[key] is not None]
        )

        if norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type == "layer":
            x = self.norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)

        return x


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()

        assert kernel_width % 2, "kernel_width must be odd."
        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
            dilation=(1, 1),
            stride=(1, 1),
            padding=(0, (kernel_width - 1) // 2),
            groups=1,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(channels * width)

        self.channels = channels
        self.width = width

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        B, C, T = inputs.shape  # BCT

        # BCT -> BcwT
        x = inputs.reshape(B, self.channels, self.width, T)
        x = self.conv2d(x)
        x = self.relu(x)

        x = x.reshape(B, C, -1)  # BcwT -> BCT


        # Skip connection after downsampling
        T_out = x.shape[-1]
        x = x + inputs[..., -T_out:]

        # Layer norm over C
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)

        return x


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        x = x.swapaxes(-1, -2)  # BCT -> BTC
        x = self.fc_block(x)
        x = x.swapaxes(-1, -2)  # BTC -> BCT
        x += inputs

        # Layer norm over C
        x = self.layer_norm(x.swapaxes(-1, -2)).swapaxes(-1, -2)
        return x


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()
        self.kernel_width = kernel_width
        self.num_blocks = len(block_channels)

        assert len(block_channels) > 0
        tds_conv_blocks = []
        for channels in block_channels:
            feature_width = num_features // channels
            assert (
                num_features % channels == 0
            ), f"block_channels {channels} must evenly divide num_features {num_features}"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, feature_width, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.tds_conv_blocks(inputs)
        return out  # (T, N, num_features)


##################
#  Tcw Encoder   #
##################


class TcwEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        in_conv_kernel_width: int = 5,
        in_conv_stride: int = 1,
        in_conv_padding: int = 2,
        num_blocks: int = 1,
        channels: int = 8,
        feature_width: int = 2,
        kernel_width: int = 1,
        out_channels: int | None = None,
    ):
        super().__init__()
        """
        Follows the multi-stage network construction from
        https://arxiv.org/abs/1904.02619.
        """

        layers: collections.OrderedDict[str, nn.Module] = collections.OrderedDict()

        C = channels * feature_width

        self.out_channels = out_channels

        # Conv1d block
        if in_conv_kernel_width > 0:
            layers["conv1dblock"] = Conv1dBlock(
                in_channels,
                C,
                kernel_size=in_conv_kernel_width,
                stride=in_conv_stride,
                padding=in_conv_padding,
            )
        elif in_channels != C:
            raise ValueError(
                f"in_channels ({in_channels}) must equal channels *"
                f" feature_width ({channels} * {feature_width}) if"
                " in_conv_kernel_width is not positive."
            )

        # TDS block
        layers["tds_block"] = TDSConvEncoder(
            num_features=C,
            block_channels=[channels] * num_blocks,
            kernel_width=kernel_width,
        )

        # Linear projection
        if out_channels is not None:
            self.linear_layer = nn.Linear(channels * feature_width, out_channels)

        self.layers = nn.Sequential(layers)

    def forward(self, x):
        x = self.layers(x)
        if self.out_channels is not None:
            x = self.linear_layer(x.swapaxes(-1, -2)).swapaxes(-1, -2)

        return x



class PianoKPMNetwork_multimodal(nn.Module):
    def __init__(
        self, pose_conv_blocks: Sequence[Conv1dBlock], key_conv_blocks: Sequence[Conv1dBlock], tds_stages: Sequence[TcwEncoder]
    ):
        super().__init__()
        self.pose_conv = nn.Sequential(*pose_conv_blocks)
        self.key_conv = nn.Sequential(*key_conv_blocks)
        self.tds_stages = nn.Sequential(*tds_stages)

        self.pose_tds = nn.Sequential(*pose_conv_blocks, *tds_stages)
        self.key_tds = nn.Sequential(*key_conv_blocks, *tds_stages)

        self.left_context = 0
        self.right_context = 0

    def forward(self, pose, key):
        seconds = (pose.shape[-1]) / 60
        n_time = round(seconds * 1024)
        pose = interpolate(pose, n_time, mode="linear", align_corners=True)

        pose_features = self.pose_conv(pose)
        key_features = self.key_conv(key)

        fused_features = pose_features + key_features

        output = self.tds_stages(fused_features)

        return output


############
# DECODERS #
############


class MLP(nn.Module):
    """Basic MLP with optional scaling of the final output."""

    def __init__(
        self,
        in_channels: int,
        layer_sizes: list[int],
        out_channels: int,
        layer_norm: bool = False,
        scale: float = 1.0,
    ):
        super().__init__()

        sizes = [in_channels] + layer_sizes
        layers = []
        for in_size, out_size in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            if layer_norm:
                layers.append(nn.LayerNorm(out_size))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(sizes[-1], out_channels))

        self.mlp = nn.Sequential(*layers)
        self.scale = scale

    def forward(self, x):
        # x is (batch, channel)
        return self.mlp(x) * self.scale

