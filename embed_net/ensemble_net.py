import jax
import jax.numpy as jnp
from jax import random, lax, vmap
from jax.nn.initializers import orthogonal, glorot_normal, he_normal
from jax.nn.functions import gelu
from functools import partial
import objax
from objax.variable import TrainVar


def _conv_block(stride, with_non_linearity, inp, kernel, bias):
    no_dilation = (1, 1)
    some_height_width = 10  # values don't matter; just shape of input
    input_shape = (1, some_height_width, some_height_width, 3)
    kernel_shape = (3, 3, 1, 1)
    input_kernel_output = ('NHWC', 'HWIO', 'NHWC')
    conv_dimension_numbers = lax.conv_dimension_numbers(input_shape,
                                                        kernel_shape,
                                                        input_kernel_output)
    block = lax.conv_general_dilated(inp, kernel, (stride, stride),
                                     'VALID', no_dilation, no_dilation,
                                     conv_dimension_numbers)
    if bias is not None:
        block += bias
    if with_non_linearity:
        block = gelu(block)
    return block


def _conv_block_without_bias(stride, with_non_linearity, inp, kernel):
    # the need for this method feels a bit clunky :/ is there a better
    # way to vmap with the None?
    return _conv_block(stride, with_non_linearity, inp, kernel, None)


class EnsembleNet(objax.Module):

    def __init__(self, num_models, dense_kernel_size=32, embedding_dim=32,
                 seed=0, logit_temp=1.0, orthogonal_init=True):

        if num_models <= 1:
            raise Exception("requires at least two models")

        self.num_models = num_models
        self.logit_temp = logit_temp

        key = random.PRNGKey(seed)
        subkeys = random.split(key, 8)

        # conv stack kernels and biases
        if orthogonal_init:
            initialiser = orthogonal
        else:
            initialiser = he_normal
        self.conv_kernels = objax.ModuleList()
        self.conv_biases = objax.ModuleList()
        input_channels = 3
        for i, output_channels in enumerate([32, 64, 64, 64, 64, 64]):
            self.conv_kernels.append(TrainVar(initialiser()(
                subkeys[i], (num_models, 3, 3, input_channels,
                             output_channels))))
            self.conv_biases.append(
                TrainVar(jnp.zeros((num_models, output_channels))))
            input_channels = output_channels

        # dense kernels and biases
        self.dense_kernels = TrainVar(initialiser()(
            subkeys[6], (num_models, 1, 1, output_channels, dense_kernel_size)))
        self.dense_biases = TrainVar(
            jnp.zeros((num_models, dense_kernel_size)))

        # embeddings kernel; no bias or non linearity.
        if orthogonal_init:
            initialiser = orthogonal
        else:
            initialiser = glorot_normal
        self.embedding_kernels = TrainVar(initialiser()(
            subkeys[7], (num_models, 1, 1, dense_kernel_size, embedding_dim)))

    def __call__(self, inp):
        # the first call vmaps over the first conv params for a single input
        y = vmap(partial(_conv_block, 2, True, inp))(
            self.conv_kernels[0].value, self.conv_biases[0].value)

        # subsequent calls vmap over both the prior input and the conv params
        # the first representing the batched input with the second representing
        # the batched models (i.e. the ensemble)
        for conv_kernel, conv_bias in zip(self.conv_kernels[1:],
                                          self.conv_biases[1:]):
            y = vmap(partial(_conv_block, 2, True))(
                y, conv_kernel.value, conv_bias.value)

        # fully convolutional dense layer (with non linearity) as bottleneck
        y = vmap(partial(_conv_block, 1, True))(
            y, self.dense_kernels.value, self.dense_biases.value)

        # final projection to embedding dim (with no activation and no bias)
        embeddings = vmap(partial(_conv_block_without_bias, 1, False))(
            y, self.embedding_kernels.value)

        # embeddings are squeezed and unit length normalised.
        embeddings = jnp.squeeze(embeddings)  # (M, N, E)
        embedding_norms = jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
        return embeddings / embedding_norms

    def calc_sims(self, crops_t0, crops_t1):
        embeddings_t0 = self(crops_t0)
        embeddings_t1 = self(crops_t1)
        model_sims = jnp.einsum('mae,mbe->ab', embeddings_t0, embeddings_t1)
        avg_model_sims = model_sims / self.num_models
        return avg_model_sims

    def loss(self, crops_t0, crops_t1, labels):
        logits_from_sims = self.calc_sims(crops_t0, crops_t1)
        logits_from_sims /= self.logit_temp
        batch_softmax_cross_entropy = jnp.mean(
            -jnp.sum(jax.nn.log_softmax(logits_from_sims) * labels, axis=-1))
        return batch_softmax_cross_entropy
