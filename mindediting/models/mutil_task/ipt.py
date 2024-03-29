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

import copy
import math

import mindspore as ms
import mindspore.ops as ops
import numpy as np
from mindspore import nn
from mindspore.common import Parameter, Tensor

from mindediting.utils.dataset_utils import add_mean, sub_mean


class MultiheadAttention(nn.Cell):
    """
    Apply multi-headed attention from "from_tensor" to "to_tensor".

    Args:
        batch_size [int]: Batch size of input datasets.
        from_tensor_width [int]: Size of last dim of from_tensor.
        to_tensor_width [int]: Size of last dim of to_tensor.
        from_seq_length [int]: Length of from_tensor sequence.
        to_seq_length [int]: Length of to_tensor sequence.
        num_attention_heads [int]: Number of attention heads. Default: 1.
        size_per_head [int]: Size of each attention head. Default: 512.
        query_act [str]: Activation function for the query transform. Default: None.
        key_act [str]: Activation function for the key transform. Default: None.
        value_act [str]: Activation function for the value transform. Default: None.
        has_attention_mask [bool]: Specifies whether to use attention mask. Default: False.
        attention_probs_dropout_prob [float]: The dropout probability for
                                      MultiheadAttention. Default: 0.0.
        use_one_hot_embeddings [bool]: Specifies whether to use one hot encoding form. Default: False.
        initializer_range [float]: Initialization value of TruncatedNormal. Default: 0.02.
        do_return_2d_tensor [bool]: True for return 2d tensor. False for return 3d
                             tensor. Default: False.
        compute_type [:class:`mindspore.dtype`]: Compute type in MultiheadAttention. Default: ms.float16.
    """

    def __init__(
        self,
        q_tensor_width,
        k_tensor_width,
        v_tensor_width,
        hidden_width,
        out_tensor_width,
        num_attention_heads=1,
        query_act=None,
        key_act=None,
        value_act=None,
        out_act=None,
        has_attention_mask=True,
        attention_probs_dropout_prob=0.0,
        use_one_hot_embeddings=False,
        initializer_range=0.02,
        do_return_2d_tensor=False,
        compute_type=ms.float16,
        same_dim=True,
    ):
        super(MultiheadAttention, self).__init__()
        self.has_attention_mask = has_attention_mask
        self.num_attention_heads = num_attention_heads
        self.size_per_head = int(hidden_width / num_attention_heads)
        self.initializer_range = initializer_range
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.do_return_2d_tensor = do_return_2d_tensor
        self.same_dim = same_dim
        input_data = [1.0 / math.sqrt(float(self.size_per_head))]
        self.scores_mul = Tensor(input_data, dtype=compute_type)
        self.reshape = ops.Reshape()
        self.shape_k_2d = (-1, k_tensor_width)
        self.shape_q_2d = (-1, q_tensor_width)
        self.shape_v_2d = (-1, v_tensor_width)
        self.hidden_width = int(hidden_width)
        if self.same_dim:
            default_input = Tensor(np.random.rand(hidden_width * 3, q_tensor_width), dtype=ms.float32)
            self.in_proj_layer = Parameter(default_input, name="weight")
        else:
            self.key_layer = nn.Dense(k_tensor_width, hidden_width, activation=key_act, has_bias=False).to_float(
                compute_type
            )
            self.query_layer = nn.Dense(q_tensor_width, hidden_width, activation=query_act, has_bias=False).to_float(
                compute_type
            )
            self.value_layer = nn.Dense(q_tensor_width, hidden_width, activation=value_act, has_bias=False).to_float(
                compute_type
            )

        self.batch_matmul_trans_b = ops.BatchMatMul(transpose_b=True)
        self.multiply = ops.Mul()
        self.transpose = ops.Transpose()
        self.trans_shape = (0, 2, 1, 3)
        self.trans_shape_relative = (2, 0, 1, 3)
        self.trans_shape_position = (1, 2, 0, 3)

        self.out_proj = nn.Dense(hidden_width, out_tensor_width, activation=out_act, has_bias=False).to_float(
            compute_type
        )

        self.multiply_data = Tensor(
            [
                -10000.0,
            ],
            dtype=compute_type,
        )
        self.batch_matmul = ops.BatchMatMul()

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(1.0 - attention_probs_dropout_prob)
        self.use_dropout = attention_probs_dropout_prob > 0

        if self.has_attention_mask:
            self.expand_dims = ops.ExpandDims()
            self.sub = ops.Sub()
            self.add = ops.Add()
            self.cast = ops.Cast()
            self.get_dtype = ops.DType()

        self.softmax_cast = ops.Cast()
        self.matmul = ops.MatMul(transpose_b=True)
        self.split = ops.Split(0, 3)
        self.equal = ops.Equal()
        self.shape = ops.Shape()

    def construct(self, tensor_q, tensor_k, tensor_v):
        """Apply multihead attention."""
        batch_size, seq_length, _ = self.shape(tensor_q)
        shape_qkv = (batch_size, -1, self.num_attention_heads, self.size_per_head)
        shape_linear = (batch_size * seq_length, self.num_attention_heads * self.size_per_head)
        if not self.do_return_2d_tensor is True:
            shape_return = (batch_size, seq_length, self.num_attention_heads * self.size_per_head)
        else:
            shape_return = (batch_size * seq_length, self.num_attention_heads * self.size_per_head)
            if seq_length == -1:
                shape_return = (-1, self.num_attention_heads * self.size_per_head)
        tensor_k_2d = self.reshape(tensor_k, self.shape_k_2d)
        tensor_q_2d = self.reshape(tensor_q, self.shape_q_2d)
        tensor_v_2d = self.reshape(tensor_v, self.shape_v_2d)

        if self.equal(tensor_q_2d, tensor_v_2d) is True:
            x = self.matmul(self.in_proj_layer, tensor_q_2d)
            query_out, key_out, value_out = self.split(x)
        elif self.same_dim is True:
            _begin = 0
            _end = self.hidden_width
            _w = self.in_proj_layer[_begin:_end, :]
            query_out = self.matmul(_w, tensor_q_2d)
            _begin = self.hidden_width
            _end = self.hidden_width * 2
            _w = self.in_proj_layer[_begin:_end, :]
            key_out = self.matmul(_w, tensor_k_2d)
            _begin = self.hidden_width * 2
            _end = None
            _w = self.in_proj_layer[_begin:]
            value_out = self.matmul(_w, tensor_v_2d)
        else:
            key_out = self.key_layer(tensor_k_2d)
            query_out = self.query_layer(tensor_q_2d)
            value_out = self.value_layer(tensor_v_2d)
        key_out = self.transpose(key_out, (1, 0))
        key_layer = self.reshape(key_out, shape_qkv)
        key_layer = self.transpose(key_layer, self.trans_shape)

        value_out = self.transpose(value_out, (1, 0))
        value_layer = self.reshape(value_out, shape_qkv)
        value_layer = self.transpose(value_layer, self.trans_shape)

        query_out = self.transpose(query_out, (1, 0))
        query_layer = self.reshape(query_out, shape_qkv)
        query_layer = self.transpose(query_layer, self.trans_shape)

        attention_scores = self.batch_matmul_trans_b(query_layer, key_layer)
        attention_scores = self.multiply(attention_scores, self.scores_mul)

        attention_scores = self.softmax_cast(attention_scores, ms.float32)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.softmax_cast(attention_probs, ms.float16)

        if self.use_dropout:
            attention_probs = self.dropout(attention_probs)

        layer_context = self.batch_matmul(attention_probs, value_layer)

        layer_context = self.transpose(layer_context, self.trans_shape)
        layer_context = self.reshape(layer_context, shape_linear)

        layer_context = self.out_proj(layer_context)
        layer_context = self.reshape(layer_context, shape_return)
        return layer_context


class LayerPreprocess(nn.Cell):
    def __init__(self, in_channels=None):
        super(LayerPreprocess, self).__init__()
        self.cast = ops.Cast()
        self.dtype = ops.DType()
        self.layernorm = nn.LayerNorm((in_channels,))

    def construct(self, input_tensor):
        output = self.cast(input_tensor, ms.float32)
        output = self.layernorm(output)
        output = self.cast(output, self.dtype(input_tensor))
        return output


class TransformerEncoderLayer(nn.Cell):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, compute_type=ms.float16):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            q_tensor_width=d_model,
            k_tensor_width=d_model,
            v_tensor_width=d_model,
            hidden_width=d_model,
            out_tensor_width=d_model,
            num_attention_heads=nhead,
            attention_probs_dropout_prob=dropout,
        )
        self.linear1 = nn.Dense(d_model, dim_feedforward).to_float(compute_type)
        self.linear2 = nn.Dense(dim_feedforward, d_model)
        self.dropout = nn.Dropout(1.0 - dropout)
        self.dropout1 = nn.Dropout(1.0 - dropout)
        self.dropout2 = nn.Dropout(1.0 - dropout)

        self.norm1 = LayerPreprocess(d_model)
        self.norm2 = LayerPreprocess(d_model)

        self.reshape = ops.Reshape()
        self.relu = ops.ReLU()

    def construct(self, src, pos=None):
        b, n, d = src.shape
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        permute_linear = (b * n, d)
        src2 = self.reshape(src2, permute_linear)
        src2 = self.linear2(self.dropout(self.relu(self.linear1(src2))))
        permute_recover = (b, n, d)
        src2 = self.reshape(src2, permute_recover)
        src = src + self.dropout2(src2)
        return src

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos


class TransformerDecoderLayer(nn.Cell):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(
            q_tensor_width=d_model,
            k_tensor_width=d_model,
            v_tensor_width=d_model,
            hidden_width=d_model,
            out_tensor_width=d_model,
            num_attention_heads=nhead,
            attention_probs_dropout_prob=dropout,
        )
        self.multihead_attn = MultiheadAttention(
            q_tensor_width=d_model,
            k_tensor_width=d_model,
            v_tensor_width=d_model,
            hidden_width=d_model,
            out_tensor_width=d_model,
            num_attention_heads=nhead,
            attention_probs_dropout_prob=dropout,
        )
        self.linear1 = nn.Dense(d_model, dim_feedforward)
        probability = 1.0 - dropout
        self.dropout = nn.Dropout(probability)
        self.linear2 = nn.Dense(dim_feedforward, d_model)

        self.dropout1 = nn.Dropout(probability)
        self.dropout2 = nn.Dropout(probability)
        self.dropout3 = nn.Dropout(probability)
        self.norm1 = LayerPreprocess(d_model)
        self.norm2 = LayerPreprocess(d_model)
        self.norm3 = LayerPreprocess(d_model)
        self.reshape = ops.Reshape()
        self.relu = ops.ReLU()

    def construct(self, tgt, memory, pos=None, query_pos=None):
        b, n, d = tgt.shape
        _tgt = self.norm1(tgt)
        q = k = self.with_pos_embed(_tgt, query_pos)
        _tgt = self.self_attn(q, k, tensor_v=_tgt)
        tgt = tgt + self.dropout1(_tgt)
        _tgt = self.norm2(tgt)
        _tgt = self.multihead_attn(
            tensor_q=self.with_pos_embed(_tgt, query_pos), tensor_k=self.with_pos_embed(memory, pos), tensor_v=memory
        )
        tgt = tgt + self.dropout2(_tgt)
        _tgt = self.norm3(tgt)
        permute_linear = (b * n, d)
        _tgt = self.reshape(_tgt, permute_linear)
        _tgt = self.linear2(self.dropout(self.relu(self.linear1(_tgt))))
        permute_recover = (b, n, d)
        _tgt = self.reshape(_tgt, permute_recover)
        tgt = tgt + self.dropout3(_tgt)
        return tgt

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos


def _get_clones(module, n):
    return nn.CellList([copy.deepcopy(module) for i in range(n)])


class TransformerEncoder(nn.Cell):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def construct(self, x, pos=None):
        output = x
        for layer in self.layers:
            output = layer(output, pos=pos)
        return output


class TransformerDecoder(nn.Cell):
    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def construct(self, x, memory, pos=None, query_pos=None):
        output = x
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)
        return output


class LearnedPositionalEncoding(nn.Cell):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length
        self.reshape = ops.Reshape()
        self.position_ids = Tensor(np.arange(self.seq_length).astype(np.int32))
        self.position_ids = self.reshape(self.position_ids, (1, self.seq_length))

    def construct(self, location_ids=None):
        if location_ids is None:
            location_ids = self.position_ids[:, : self.seq_length]
        position_embeddings = self.pe(location_ids)
        return position_embeddings


class VisionTransformer(nn.Cell):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        num_queries,
        dropout_rate=0,
        norm=False,
        mlp=False,
        pos_every=False,
        no_pos=False,
        con_loss=False,
    ):
        super(VisionTransformer, self).__init__()
        self.norm = norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.pos_every = pos_every
        self.num_channels = num_channels
        self.img_dim = img_dim
        self.patch_dim = patch_dim
        self.num_patches = int((img_dim // patch_dim) ** 2)
        dim_value = patch_dim * patch_dim * num_channels
        self.flatten_dim = dim_value
        self.out_dim = dim_value
        self.no_pos = no_pos
        self.unf = _unfold_(patch_dim)
        self.fold = _fold_(patch_dim, output_shape=(img_dim, img_dim))

        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        self.seq_length = self.num_patches
        self.query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)

        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        keep_prob = 1.0 - dropout_rate
        if self.mlp is not True:
            self.mlp_head = nn.SequentialCell(
                nn.Dense(embedding_dim, hidden_dim),
                nn.Dropout(keep_prob),
                nn.ReLU(),
                nn.Dense(hidden_dim, self.out_dim),
                nn.Dropout(keep_prob),
            )
            self.linear_encoding = nn.Dense(self.flatten_dim, embedding_dim)
        self.reshape = ops.Reshape()
        self.tile = ops.Tile()
        self.transpose = ops.Transpose()
        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(self.seq_length, self.embedding_dim, self.seq_length)

        self.dropout_layer1 = nn.Dropout(keep_prob)
        self.con_loss = con_loss

    def construct(self, x, query_idx_tensor):
        x = self.unf(x)
        b, n, _ = x.shape

        if not self.mlp is True:
            x = self.reshape(x, (b * n, -1))
            x = self.dropout_layer1(self.linear_encoding(x)) + x
            x = self.reshape(x, (b, n, -1))

        if self.no_pos:
            x = self.encoder(x)
        else:
            pos = self.position_encoding()
            x = self.encoder(x + pos)
        query_embed = self.tile(
            self.reshape(self.query_embed(query_idx_tensor), (1, self.seq_length, self.embedding_dim)), (b, 1, 1)
        )
        x = self.decoder(x, x, query_pos=query_embed)

        if not self.mlp is True:
            x = self.reshape(x, (b * n, -1))
            x = self.mlp_head(x) + x
            x = self.reshape(x, (b, n, -1))
        if self.con_loss:
            conx = x
            x = self.fold(x)
            return x, conx
        x = self.fold(x)
        return x


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.reshape = ops.Reshape()
        self.eye = ops.Eye()
        std = Tensor(rgb_std, ms.float32)
        self.weight.set_data(self.reshape(self.eye(3, 3, ms.float32), (3, 3, 1, 1)) / self.reshape(std, (3, 1, 1, 1)))
        self.weight.requires_grad = False
        self.bias = Parameter(sign * rgb_range * Tensor(rgb_mean, ms.float32) / std, name="bias", requires_grad=False)
        self.has_bias = True


def default_conv(in_channels, out_channels, kernel_size, has_bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, has_bias=has_bias)


class ResBlock(nn.Cell):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(), res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        layers = []
        for i in range(2):
            layers.append(conv(n_feats, n_feats, kernel_size, has_bias=bias))
            if bn:
                layers.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                layers.append(act)
        self.mul = ops.Mul()
        self.body = nn.SequentialCell(*layers)

    def construct(self, x):
        res = self.mul(self.body(x), self.res_scale)
        res += x
        return res


class SmallUpSampler(nn.Cell):
    def __init__(self, conv, upsize, n_feats, bias=True):
        super(SmallUpSampler, self).__init__()
        self.conv = conv(n_feats, upsize * upsize * n_feats, 3, bias)
        self.reshape = ops.Reshape()
        self.upsize = upsize

    def construct(self, _input):
        _input = self.conv(_input)
        output = self.pixelsf(_input, self.upsize)
        return output

    @staticmethod
    def pixelsf(x, scale):
        n, c, ih, iw = x.shape
        oh = ih * scale
        ow = iw * scale
        oc = c // (scale**2)
        output = ops.Transpose()(x, (0, 2, 1, 3))
        output = ops.Reshape()(output, (n, ih, oc * scale, scale, iw))
        output = ops.Transpose()(output, (0, 1, 2, 4, 3))
        output = ops.Reshape()(output, (n, ih, oc, scale, ow))
        output = ops.Transpose()(output, (0, 2, 1, 3, 4))
        output = ops.Reshape()(output, (n, oc, oh, ow))
        return output


class Upsampler(nn.Cell):
    def __init__(self, conv, scale, n_feats, bias=True):
        super(Upsampler, self).__init__()
        layers = []
        if (scale & (scale - 1)) == 0:
            [layers.append(SmallUpSampler(conv, 2, n_feats, bias=bias)) for _ in range(int(math.log(scale, 2)))]
        elif scale == 3:
            layers.append(SmallUpSampler(conv, 3, n_feats, bias=bias))
        self.net = nn.SequentialCell(layers)

    def construct(self, x):
        return self.net(x)


class IPT(nn.Cell):
    def __init__(self, args):
        super(IPT, self).__init__()
        self.dytpe = ms.float16
        self.scale_idx = 0
        self.args = args
        self.con_loss = args.loss.con_loss
        self.test_only = args.model.test_only

        self.reshape = ops.Reshape()
        self.tile = ops.Tile()
        self.transpose = ops.Transpose()
        self.scalar_to_tensor = ops.ScalarToTensor()
        self.cast = ops.Cast()

        kernel_size = 3
        act = nn.ReLU()
        conv = default_conv

        self.head = nn.CellList(
            [
                nn.SequentialCell(
                    conv(args.model.n_colors, args.model.n_feats, kernel_size).to_float(self.dytpe),
                    ResBlock(conv, args.model.n_feats, 5, act=act).to_float(self.dytpe),
                    ResBlock(conv, args.model.n_feats, 5, act=act).to_float(self.dytpe),
                )
                for _ in args.dataset.scale
            ]
        )
        embedding_dim = args.model.n_feats * args.model.transformer.patch_dim**2
        self.body = VisionTransformer(
            img_dim=args.dataset.patch_size,
            patch_dim=args.model.transformer.patch_dim,
            num_channels=args.model.n_feats,
            embedding_dim=embedding_dim,
            num_heads=args.model.transformer.num_heads,
            num_layers=args.model.transformer.num_layers,
            hidden_dim=embedding_dim * 4,
            num_queries=args.model.transformer.num_queries,
            dropout_rate=args.model.transformer.dropout_rate,
            mlp=args.model.transformer.no_mlp,
            pos_every=args.model.transformer.pos_every,
            no_pos=args.model.transformer.no_pos,
            con_loss=args.loss.con_loss,
        ).to_float(self.dytpe)

        self.tail = nn.CellList(
            [
                nn.SequentialCell(
                    Upsampler(conv, s, args.model.n_feats).to_float(self.dytpe),
                    conv(args.model.n_feats, args.model.n_colors, kernel_size).to_float(self.dytpe),
                )
                for s in args.dataset.scale
            ]
        )

    def construct(self, x, idx):
        idx_number = idx.shape[0]

        if self.test_only:
            idx_number = idx.shape[0] * 0
        x = self.head[idx_number](x)
        idx_tensor = self.cast(self.scalar_to_tensor(idx_number), ms.int32)
        if self.con_loss:
            res, x_con = self.body(x, idx_tensor)
            res += x
            x = self.tail[idx_number](x)
            return x, x_con
        res = self.body(x, idx_tensor)
        res += x
        x = self.tail[idx_number](res)
        return x


class IPTWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn, use_con=True):
        super(IPTWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self.use_con = use_con
        if backbone.jit_config_dict:
            self._jit_config_dict = backbone.jit_config_dict

    def construct(self, lr, hr, idx):
        if self.use_con:
            out, con_loss = self._backbone(lr, idx)
            return self._loss_fn([out, con_loss], hr)
        else:
            out = self._backbone(lr, idx)
            return self._loss_fn(out, hr)

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, the backbone network.
        """
        return self._backbone


class IPT_post(nn.Cell):
    """ipt"""

    def __init__(self, model, args):
        super(IPT_post, self).__init__()
        self.idx = Tensor(np.ones(args.task.alltask.task_id), ms.int32)
        self.model = model
        self.args = args
        self.scale_idx = 0
        self.reshape = ops.Reshape()
        self.tile = ops.Tile()
        self.transpose = ops.Transpose()
        self.concat0 = ops.Concat(axis=0)
        self.concat2 = ops.Concat(axis=2)
        self.concat3 = ops.Concat(axis=3)

    def construct(self, x):
        """ipt"""
        self.model.con_loss = False
        self.model.body.con_loss = False
        x = sub_mean(x)
        batchsize = 64
        h, w = x.shape[-2:]
        dataset_config = self.args.val_dataset if hasattr(self.args, "val_dataset") else self.args.dataset
        padsize = int(dataset_config.patch_size)
        shave = int(dataset_config.patch_size * 0.25)
        scale = dataset_config.scale[0]
        h_cut = (h - padsize) % (padsize - shave)
        w_cut = (w - padsize) % (padsize - shave)

        x_hw_cut = x[:, :, (h - padsize) :, (w - padsize) :]
        y_hw_cut = self.model(x_hw_cut, self.idx)

        x_h_cut = x[:, :, (h - padsize) :, :]
        x_w_cut = x[:, :, :, (w - padsize) :]
        y_h_cut = self.cut_h_new(x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        y_w_cut = self.cut_w_new(x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)

        x_h_top = x[:, :, :padsize, :]
        x_w_top = x[:, :, :, :padsize]
        y_h_top = self.cut_h_new(x_h_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        y_w_top = self.cut_w_new(x_w_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)

        unf_1 = _stride_unfold_(padsize, stride=padsize - shave)
        x_unfold = unf_1.compute(x)
        x_unfold = self.transpose(x_unfold, (1, 0, 2))  # transpose(0,2)

        x_unfold = self.reshape(x_unfold, (x_unfold.shape[0], -1, padsize, padsize))
        x_range = x_unfold.shape[0] // batchsize + (x_unfold.shape[0] % batchsize != 0)
        for i in range(x_range):
            if i == 0:
                y_unfold = self.model(x_unfold[i * batchsize : (i + 1) * batchsize, :, :, :], self.idx)
            else:
                y_unfold = self.concat0(
                    (y_unfold, self.model(x_unfold[i * batchsize : (i + 1) * batchsize, :, :, :], self.idx))
                )
        y_unf_shape_0 = y_unfold.shape[0]
        padsize_mul_scale = padsize * scale
        fold_1 = _stride_fold_(
            padsize_mul_scale,
            output_shape=((h - h_cut) * scale, (w - w_cut) * scale),
            stride=padsize_mul_scale - shave * scale,
        )
        y = fold_1.compute(self.transpose(self.reshape(y_unfold, (y_unf_shape_0, -1, 1)), (2, 0, 1)))
        if y[:, :, padsize_mul_scale:, :].shape[2] == 0:
            y = y_h_top
        else:
            y = self.concat2((y_h_top, y[:, :, padsize_mul_scale:, :]))
        if y[:, :, :, padsize_mul_scale:].shape[3] == 0:
            y = y_w_top
        else:
            y = self.concat3((y_w_top, y[:, :, :, padsize_mul_scale:]))
        scale_half = 0.5 * scale
        _slice = int(shave * scale_half)
        y_unfold = y_unfold[:, :, _slice : padsize_mul_scale - _slice, _slice : padsize_mul_scale - _slice]
        fold_2 = _stride_fold_(
            padsize_mul_scale - shave * scale,
            output_shape=((h - h_cut - shave) * scale, (w - w_cut - shave) * scale),
            stride=padsize_mul_scale - shave * scale,
        )
        y_inter = fold_2.compute(self.transpose(self.reshape(y_unfold, (y_unf_shape_0, -1, 1)), (2, 0, 1)))
        concat1 = self.concat2((y[:, :, :_slice, _slice : (w - w_cut) * scale - _slice], y_inter))
        concat2 = self.concat2(
            (concat1, y[:, :, (h - h_cut) * scale - _slice :, _slice : (w - w_cut) * scale - _slice])
        )
        concat3 = self.concat3((y[:, :, :, :_slice], concat2))
        y = self.concat3((concat3, y[:, :, :, (w - w_cut) * scale - _slice :]))
        y = self.concat2(
            (
                y[:, :, : y.shape[2] - int((padsize - h_cut) * scale_half), :],
                y_h_cut[:, :, int((padsize - h_cut) * scale_half + 0.5) :, :],
            )
        )
        y_w_cat = self.concat2(
            (
                y_w_cut[:, :, : y_w_cut.shape[2] - int((padsize - h_cut) * scale_half), :],
                y_hw_cut[:, :, int((padsize - h_cut) * scale_half + 0.5) :, :],
            )
        )
        y = self.concat3(
            (
                y[:, :, :, : y.shape[3] - int((padsize - w_cut) * scale_half)],
                y_w_cat[:, :, :, int((padsize - w_cut) * scale_half + 0.5) :],
            )
        )

        y = add_mean(y)
        y = y / 255.0
        self.model.con_loss = True
        self.model.body.con_loss = True
        return y

    def cut_h_new(self, x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):
        unf_1 = _stride_unfold_(padsize, stride=padsize - shave)
        x_h_cut_unfold = unf_1.compute(x_h_cut)
        x_h_cut_unfold = self.transpose(x_h_cut_unfold, (1, 0, 2))
        x_h_cut_unfold = self.reshape(x_h_cut_unfold, (x_h_cut_unfold.shape[0], -1, padsize, padsize))
        x_range = x_h_cut_unfold.shape[0] // batchsize + (x_h_cut_unfold.shape[0] % batchsize != 0)
        for i in range(x_range):
            if i == 0:
                y_h_cut_unfold = self.model(x_h_cut_unfold[i * batchsize : (i + 1) * batchsize, :, :, :], self.idx)
            else:
                y_h_cut_unfold = self.concat0(
                    (y_h_cut_unfold, self.model(x_h_cut_unfold[i * batchsize : (i + 1) * batchsize, :, :, :], self.idx))
                )
        y_h_cut_unfold_shape_0 = y_h_cut_unfold.shape[0]
        padsize_mul_scale = padsize * scale
        fold_1 = _stride_fold_(
            padsize_mul_scale,
            output_shape=(padsize_mul_scale, (w - w_cut) * scale),
            stride=padsize_mul_scale - shave * scale,
        )
        y_h_cut = fold_1.compute(
            self.transpose(self.reshape(y_h_cut_unfold, (y_h_cut_unfold_shape_0, -1, 1)), (2, 0, 1))
        )
        _slice = int(shave * 0.5 * scale)
        y_h_cut_unfold = y_h_cut_unfold[:, :, :, _slice : padsize_mul_scale - _slice]
        fold_2 = _stride_fold_(
            (padsize_mul_scale, padsize_mul_scale - shave * scale),
            output_shape=(padsize_mul_scale, (w - w_cut - shave) * scale),
            stride=padsize_mul_scale - shave * scale,
        )
        y_h_cut_inter = fold_2.compute(
            self.transpose(self.reshape(y_h_cut_unfold, (y_h_cut_unfold_shape_0, -1, 1)), (2, 0, 1))
        )
        concat1 = self.concat3((y_h_cut[:, :, :, :_slice], y_h_cut_inter))
        y_h_cut = self.concat3((concat1, y_h_cut[:, :, :, (w - w_cut) * scale - _slice :]))
        return y_h_cut

    def cut_w_new(self, x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):
        unf_1 = _stride_unfold_(padsize, stride=padsize - shave)
        x_w_cut_unfold = unf_1.compute(x_w_cut)
        x_w_cut_unfold = self.transpose(x_w_cut_unfold, (1, 0, 2))
        x_w_cut_unfold = self.reshape(x_w_cut_unfold, (x_w_cut_unfold.shape[0], -1, padsize, padsize))
        x_range = x_w_cut_unfold.shape[0] // batchsize + (x_w_cut_unfold.shape[0] % batchsize != 0)
        for i in range(x_range):
            if i == 0:
                y_w_cut_unfold = self.model(x_w_cut_unfold[i * batchsize : (i + 1) * batchsize, :, :, :], self.idx)
            else:
                y_w_cut_unfold = self.concat0(
                    (y_w_cut_unfold, self.model(x_w_cut_unfold[i * batchsize : (i + 1) * batchsize, :, :, :], self.idx))
                )
        y_w_cut_unfold_shape_0 = y_w_cut_unfold.shape[0]
        padsize_mul_scale = padsize * scale
        fold_1 = _stride_fold_(
            padsize_mul_scale,
            output_shape=((h - h_cut) * scale, padsize_mul_scale),
            stride=padsize_mul_scale - shave * scale,
        )
        y_w_cut = fold_1.compute(
            self.transpose(self.reshape(y_w_cut_unfold, (y_w_cut_unfold_shape_0, -1, 1)), (2, 0, 1))
        )
        _slice = int(shave * 0.5 * scale)
        y_w_cut_unfold = y_w_cut_unfold[:, :, _slice : padsize_mul_scale - _slice, :]
        fold_2 = _stride_fold_(
            (padsize_mul_scale - shave * scale, padsize_mul_scale),
            output_shape=((h - h_cut - shave) * scale, padsize_mul_scale),
            stride=padsize_mul_scale - shave * scale,
        )
        y_w_cut_inter = fold_2.compute(
            self.transpose(self.reshape(y_w_cut_unfold, (y_w_cut_unfold_shape_0, -1, 1)), (2, 0, 1))
        )
        concat1 = self.concat2((y_w_cut[:, :, :_slice, :], y_w_cut_inter))
        y_w_cut = self.concat2((concat1, y_w_cut[:, :, (h - h_cut) * scale - _slice :, :]))
        return y_w_cut


class _stride_unfold_:
    def __init__(self, kernel_size, stride=-1):
        super(_stride_unfold_, self).__init__()
        self.kernel_size = kernel_size
        if stride == -1:
            self.stride = kernel_size
        else:
            self.stride = stride
        self.unfold = _unfold_(kernel_size)

    def compute(self, x):
        x = x.asnumpy()
        N, C, H, W = x.shape
        leftup_idx_x = []
        leftup_idx_y = []
        nh = (H - self.kernel_size) // self.stride + 1
        for i in range(nh):
            leftup_idx_x.append(i * self.stride)
        nw = (W - self.kernel_size) // self.stride + 1
        for i in range(nw):
            leftup_idx_y.append(i * self.stride)
        xNumBlock = len(leftup_idx_x)
        yNumBlock = len(leftup_idx_y)
        unf_x = np.zeros((N, C, xNumBlock * self.kernel_size, yNumBlock * self.kernel_size), dtype=np.float32)
        N, C, H, W = unf_x.shape
        for i in range(xNumBlock):
            for j in range(yNumBlock):
                unf_i = i * self.kernel_size
                unf_j = j * self.kernel_size
                unf_jkz = unf_j + self.kernel_size
                org_i = leftup_idx_x[i]
                org_j = leftup_idx_y[j]
                fills = x[:, :, org_i : org_i + self.kernel_size, org_j : org_j + self.kernel_size]
                zeros2 = np.zeros(unf_x[:, :, :unf_i, unf_j:unf_jkz].shape)
                concat1 = np.concatenate((zeros2, fills), axis=2)
                zeros3 = np.zeros(unf_x[:, :, unf_i + self.kernel_size :, unf_j:unf_jkz].shape)
                concat2 = np.concatenate((concat1, zeros3), axis=2)
                zeros1 = np.zeros(unf_x[:, :, :, :unf_j].shape)
                concat3 = np.concatenate((zeros1, concat2), axis=3)
                zeros4 = np.zeros(unf_x[:, :, :, unf_jkz:].shape)
                concat4 = np.concatenate((concat3, zeros4), axis=3)
                unf_x += concat4
        unf_x = Tensor(unf_x, ms.float16)
        y = self.unfold(unf_x)
        return y


class _stride_fold_:
    def __init__(self, kernel_size, output_shape=(-1, -1), stride=-1):
        super(_stride_fold_, self).__init__()
        if isinstance(kernel_size, (list, tuple)):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = [kernel_size, kernel_size]
        self.output_shape = output_shape
        if stride == -1:
            self.stride = kernel_size[0]
        else:
            self.stride = stride

        self.NumBlock_x = (self.output_shape[0] - self.kernel_size[0]) // self.stride + 1
        self.NumBlock_y = (self.output_shape[1] - self.kernel_size[1]) // self.stride + 1
        self.large_shape = [self.NumBlock_x * self.kernel_size[0], self.NumBlock_y * self.kernel_size[1]]
        self.fold = _fold_(self.kernel_size, self.large_shape)

    def compute(self, x):
        xNumBlock = self.NumBlock_x
        yNumBlock = self.NumBlock_y
        large_x = self.fold(x)
        large_x = large_x.asnumpy()
        N, C, _, _ = large_x.shape
        leftup_idx_x = []
        leftup_idx_y = []
        for i in range(xNumBlock):
            leftup_idx_x.append(i * self.kernel_size[0])
        for i in range(yNumBlock):
            leftup_idx_y.append(i * self.kernel_size[1])
        fold_x = np.zeros(
            (
                N,
                C,
                (xNumBlock - 1) * self.stride + self.kernel_size[0],
                (yNumBlock - 1) * self.stride + self.kernel_size[1],
            ),
            dtype=np.float32,
        )
        for i in range(xNumBlock):
            for j in range(yNumBlock):
                fold_i = i * self.stride
                fold_j = j * self.stride
                fold_jkz = fold_j + self.kernel_size[1]
                org_i = leftup_idx_x[i]
                org_j = leftup_idx_y[j]
                fills = large_x[:, :, org_i : org_i + self.kernel_size[0], org_j : org_j + self.kernel_size[1]]
                t2 = fold_x[:, :, :fold_i, fold_j:fold_jkz]
                zeros2 = np.zeros(t2.shape)
                concat1 = np.concatenate((zeros2, fills), axis=2)
                t3 = fold_x[:, :, fold_i + self.kernel_size[0] :, fold_j:fold_jkz]
                zeros3 = np.zeros(t3.shape)
                concat2 = np.concatenate((concat1, zeros3), axis=2)
                t1 = fold_x[:, :, :, :fold_j]
                zeros1 = np.zeros(t1.shape)
                concat3 = np.concatenate((zeros1, concat2), axis=3)
                t4 = fold_x[:, :, :, fold_jkz:]
                zeros4 = np.zeros(t4.shape)
                concat4 = np.concatenate((concat3, zeros4), axis=3)
                fold_x += concat4
        y = Tensor(fold_x, ms.float16)
        return y


class _unfold_(nn.Cell):
    def __init__(self, kernel_size, stride=-1):
        super(_unfold_, self).__init__()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        if stride == -1:
            self.stride = kernel_size
        self.kernel_size = kernel_size

    def construct(self, x):
        N, C, H, W = x.shape
        numH = H // self.kernel_size
        numW = W // self.kernel_size
        numw_mul_kernel_size = numW * self.kernel_size
        if numH * self.kernel_size != H or numw_mul_kernel_size != W:
            x = x[:, :, : numH * self.kernel_size, :, numw_mul_kernel_size]
        output_img = self.reshape(x, (N, C, numH, self.kernel_size, W))
        numH_mul_numW = numH * numW
        output_img = self.transpose(output_img, (0, 1, 2, 4, 3))
        output_img = self.reshape(output_img, (N * C, numH, numW, self.kernel_size, self.kernel_size))
        output_img = self.transpose(output_img, (0, 1, 2, 4, 3))
        output_img = self.reshape(output_img, (N, C, numH_mul_numW, self.kernel_size * self.kernel_size))
        output_img = self.transpose(output_img, (0, 2, 1, 3))
        output_img = self.reshape(output_img, (N, numH_mul_numW, -1))
        return output_img


class _fold_(nn.Cell):
    def __init__(self, kernel_size, output_shape=(-1, -1), stride=-1):
        super(_fold_, self).__init__()
        self.output_shape = output_shape
        if isinstance(kernel_size, (list, tuple)):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = [kernel_size, kernel_size]

        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.sqrt = ops.Sqrt()
        self.cast = ops.Cast()

        if stride == -1:
            self.stride = self.kernel_size[0]

    def construct(self, _input):
        N, C, L = _input.shape
        H, W = self.kernel_size[0], self.kernel_size[1]
        org_C = L // (H * W)
        org_H = self.output_shape[0]
        org_W = self.output_shape[1]
        numH = org_H // H
        numW = org_W // W
        output_img = self.reshape(_input, (N, C, org_C, H, W))
        output_img = self.transpose(output_img, (0, 2, 3, 1, 4))
        output_img = self.reshape(output_img, (N * org_C, H, numH, numW, W))
        output_img = self.transpose(output_img, (0, 2, 1, 3, 4))

        output_img = self.reshape(output_img, (N, org_C, org_H, org_W))
        return output_img
