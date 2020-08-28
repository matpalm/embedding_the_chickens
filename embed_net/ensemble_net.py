import jax
import jax.numpy as jnp
from jax import random, lax, vmap, jit, grad
from jax.nn.initializers import orthogonal, glorot_normal, he_normal
from jax.nn.functions import gelu
from functools import partial


def initial_params(num_models, dense_kernel_size=32, embedding_dim=32,
                   seed=0, orthogonal_init=True):

    if num_models <= 1:
        raise Exception("requires at least two models")

    key = random.PRNGKey(seed)
    subkeys = random.split(key, 8)

    # conv stack kernels and biases
    if orthogonal_init:
        initialiser = orthogonal
    else:
        initialiser = he_normal
    conv_kernels = []
    conv_biases = []
    input_channels = 3
    for i, output_channels in enumerate([32, 64, 64, 64, 64, 64]):
        conv_kernels.append(initialiser()(
            subkeys[i], (num_models, 3, 3, input_channels, output_channels)))
        conv_biases.append(jnp.zeros((num_models, output_channels)))
        input_channels = output_channels

    # dense kernels and biases
    dense_kernels = initialiser()(
        subkeys[6], (num_models, 1, 1, 64, dense_kernel_size))
    dense_biases = jnp.zeros((num_models, dense_kernel_size))

    # embeddings kernel; no bias or non linearity.
    if orthogonal_init:
        initialiser = orthogonal
    else:
        initialiser = glorot_normal
    embedding_kernels = initialiser()(
        subkeys[7], (num_models, 1, 1, dense_kernel_size, embedding_dim))

    # return all params
    return conv_kernels + conv_biases + [dense_kernels, dense_biases,
                                         embedding_kernels]


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


@jit
def embed(params, inp):
    assert len(params) == 15
    conv_kernels = params[0:6]
    conv_biases = params[6:12]
    dense_kernels = params[12]
    dense_biases = params[13]
    embedding_kernels = params[14]

    # the first call vmaps over the first conv params for a single input
    y = vmap(partial(_conv_block, 2, True, inp))(
        conv_kernels[0], conv_biases[0])

    # subsequent calls vmap over both the prior input and the conv params
    # the first representing the batched input with the second representing
    # the batched models (i.e. the ensemble)
    for conv_kernel, conv_bias in zip(conv_kernels[1:], conv_biases[1:]):
        y = vmap(partial(_conv_block, 2, True))(y, conv_kernel, conv_bias)

    # fully convolutional dense layer (with non linearity) as bottleneck
    y = vmap(partial(_conv_block, 1, True))(y, dense_kernels, dense_biases)

    # final projection to embedding dim (with no activation and no bias)
    embeddings = vmap(partial(_conv_block_without_bias, 1, False))(
        y, embedding_kernels)
    # embeddings are squeezed and unit length normalised.
    embeddings = jnp.squeeze(embeddings)  # (M, N, E)
    embedding_norms = jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / embedding_norms


@jit
def calc_sims(params, crops_t0, crops_t1):
    num_models = params[0].shape[0]
    embeddings_t0 = embed(params, crops_t0)
    embeddings_t1 = embed(params, crops_t1)
    model_sims = jnp.einsum('mae,mbe->ab', embeddings_t0, embeddings_t1)
    avg_model_sims = model_sims / num_models
    return avg_model_sims


# @jit
def loss(params, crops_t0, crops_t1, labels, temp):
    logits_from_sims = calc_sims(params, crops_t0, crops_t1)
    logits_from_sims /= temp
    batch_softmax_cross_entropy = jnp.mean(
        -jnp.sum(jax.nn.log_softmax(logits_from_sims) * labels, axis=-1))
    return batch_softmax_cross_entropy
