# Copyright © 2023 Huawei Technologies Co, Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from mindspore import nn, ops


class Convolution(nn.Cell):
    def __init__(self, in_channels=64, num_filters=64, kernel_size=3, strides=1, use_bias=True, use_bn=False):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, num_filters, kernel_size, strides, pad_mode="same", has_bias=use_bias))
        if use_bn:
            layers.append(nn.BatchNorm2d(num_filters))
        layers.append(nn.ReLU())
        self.conv = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.conv(x)
        return output


class ResBlock(nn.Cell):
    def __init__(self, in_channels=64, num_filters=64, kernel_size=3, with_shortcut=True, use_bias=True, use_bn=False):
        super().__init__()
        self.conv0 = Convolution(in_channels, num_filters, kernel_size, use_bn=use_bn)
        self.conv1 = Convolution(num_filters, in_channels, kernel_size, use_bn=use_bn)

    def construct(self, x):
        out = x
        out = self.conv0(out)
        out = self.conv1(out)
        out = x + out
        return out


class DownsampleLayer(nn.Cell):
    def __init__(self, factor, in_channels, num_filters=64, kernel_size=3):
        super().__init__()

        self.factor = factor

        self.downsample = nn.Conv2d(
            in_channels, num_filters, kernel_size, stride=factor, pad_mode="same", has_bias=True
        )

    def construct(self, x):
        if x is None or self.factor == 1:
            return x
        out = self.downsample(x)
        return out


class UpsampleLayer(nn.Cell):
    def __init__(self, factor, in_channels, num_filters=64, kernel_size=2):
        super().__init__()

        self.upsample = nn.Conv2dTranspose(
            in_channels, num_filters, kernel_size, pad_mode="same", stride=factor, has_bias=True
        )

    def construct(self, x):
        out = self.upsample(x)
        return out


class Unet(nn.Cell):
    def __init__(
        self,
        num_enc_blocks=None,
        num_enc_filters=None,
        enc_kernel_size=3,
        num_dec_blocks=None,
        num_dec_filters=None,
        dec_kernel_size=3,
        with_batch_norm=False,
    ):
        super().__init__()
        if num_enc_blocks is None:
            num_enc_blocks = [1, 1, 1]
        if num_enc_filters is None:
            num_enc_filters = [16, 16, 32]
        if num_dec_blocks is None:
            num_dec_blocks = [1, 1, 1]
        if num_dec_filters is None:
            num_dec_filters = [16, 16, 32]

        self.num_levels = len(num_enc_blocks)
        # encoder
        self.enc_chain = nn.CellList()
        self.dec_chain = nn.CellList()
        for i in range(self.num_levels):
            enc_ops = []
            if i > 0:
                enc_ops.append(
                    DownsampleLayer(factor=2, in_channels=num_enc_filters[i - 1], num_filters=num_enc_filters[i - 1])
                )
            for j in range(num_enc_blocks[i]):
                enc_ops.append(
                    ResBlock(
                        in_channels=(num_enc_filters[i - 1] if i > 0 else num_enc_filters[i]),
                        num_filters=num_enc_filters[i],
                        kernel_size=enc_kernel_size,
                        use_bn=with_batch_norm,
                    )
                )
            self.enc_chain.append(nn.SequentialCell(enc_ops))
        # decoder
        self.upsampling = nn.CellList()
        self.dec_chain = nn.CellList()
        for i in range(self.num_levels - 1, 0, -1):
            self.upsampling.append(
                UpsampleLayer(
                    factor=2,
                    in_channels=num_dec_filters[i - 1],
                    num_filters=num_dec_filters[i - 1],
                    kernel_size=dec_kernel_size,
                )
            )
            dec_ops = []
            dec_ops.append(
                Convolution(
                    in_channels=2 * num_dec_filters[i - 1],
                    num_filters=num_dec_filters[i - 1],
                    kernel_size=dec_kernel_size,
                    use_bn=with_batch_norm,
                )
            )

            for j in range(num_dec_blocks[i]):
                dec_ops.append(
                    ResBlock(
                        in_channels=num_dec_filters[i - 1],
                        num_filters=num_dec_filters[i - 1],
                        kernel_size=dec_kernel_size,
                        use_bn=with_batch_norm,
                    )
                )
            self.dec_chain.append(nn.SequentialCell(dec_ops))

    def construct(self, x):
        out = x
        skip_connections = []
        for i in range(self.num_levels):
            out = self.enc_chain[i](out)
            skip_connections.append(out)
        for i in range(self.num_levels - 1, 0, -1):
            out = self.upsampling[i - 1](out)
            conc = ops.Concat(1)([out, skip_connections[i - 1]])
            out = self.dec_chain[i - 1](conc)
        return out


class NoahTCVNet(nn.Cell):
    def __init__(self, use_bn=False):
        super().__init__()

        self.num_channels = 3
        self.num_fore_filter = 16
        self.num_fore_blocks = 1
        self.num_last_filter = 16
        self.num_last_blocks = 1

        self.num_enc_blocks = [1, 1, 1]
        self.num_dec_blocks = [1, 1, 1]
        self.enc_kernel_size = 3
        self.dec_kernel_size = 3
        self.num_enc_filters = [16, 16, 32]
        self.num_dec_filters = [16, 16, 32]

        assert len(self.num_enc_blocks) == len(self.num_enc_filters)
        assert len(self.num_enc_blocks) > 0
        assert len(self.num_dec_blocks) == len(self.num_dec_filters)
        assert len(self.num_dec_blocks) > 0
        assert self.num_fore_filter == self.num_enc_filters[0]
        assert self.num_last_filter == self.num_dec_filters[0]

        # fore_conv
        self.fore_conv = Convolution(self.num_channels, self.num_fore_filter)

        # backbone
        self.backbone = Unet(
            num_enc_blocks=self.num_enc_blocks,
            num_enc_filters=self.num_enc_filters,
            enc_kernel_size=self.enc_kernel_size,
            num_dec_blocks=self.num_dec_blocks,
            num_dec_filters=self.num_dec_filters,
            dec_kernel_size=self.dec_kernel_size,
            with_batch_norm=use_bn,
        )

        # final feature reconstruction
        self.last_blocks = Convolution(self.num_last_filter, self.num_last_filter)

        # output
        self.outConv = nn.Conv2d(self.num_last_filter, self.num_channels, 3, pad_mode="same", has_bias=True)

    def construct(self, x):
        # initial feature extraction with num_filters corresponding to initial level

        # fore conv
        output = self.fore_conv(x)

        # backbone
        output = self.backbone(output)

        # final feature reconstruction
        output = self.last_blocks(output)

        # output
        out = self.outConv(output)

        out = out + x
        return out
