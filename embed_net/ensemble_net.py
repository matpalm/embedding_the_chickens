import jax
import jax.numpy as jnp
from jax import random, lax, vmap, jit
from jax.nn.initializers import orthogonal
from jax.nn.functions import relu
from functools import partial


def conv_block(stride, with_relu, input, kernel):
    no_dilation = (1, 1)
    some_height_width = 10  # values don't matter; just shape of input
    input_shape = (1, some_height_width, some_height_width, 3)
    kernel_shape = (3, 3, 1, 1)
    input_kernel_output = ('NHWC', 'HWIO', 'NHWC')
    conv_dimension_numbers = lax.conv_dimension_numbers(input_shape,
                                                        kernel_shape,
                                                        input_kernel_output)
    block = lax.conv_general_dilated(input, kernel, (stride, stride),
                                     'VALID', no_dilation, no_dilation,
                                     conv_dimension_numbers)
    if with_relu:
        block = relu(block)
    return block


def initial_params(num_models):
    key = random.PRNGKey(0)
    subkeys = random.split(key, 8)

    # conv stack
    conv_kernels = []
    input_size = 3
    for i, output_size in enumerate([32, 64, 64, 64, 64, 64]):
        conv_kernels.append(orthogonal()(
            subkeys[i], (num_models, 3, 3, input_size, output_size)))
        input_size = output_size
    # dense kernel
    dense_kernels = orthogonal()(subkeys[6], (num_models, 1, 1, 64, 32))
    # embeddings
    embedding_kernels = orthogonal()(subkeys[7], (num_models, 1, 1, 32, 32))

    # return all params
    return conv_kernels + [dense_kernels, embedding_kernels]


@jit
def embed(params, input):
    assert len(params) == 8
    conv_kernels = params[0:6]
    dense_kernels = params[6]
    embedding_kernels = params[7]

    # the first call vmaps over the first kernels for a single input
    y = vmap(partial(conv_block, 2, True, input))(conv_kernels[0])

    # subsequent calls vmap over both the prior input and the kernel
    # the first representing the batched input with the second representing
    # the batched models (i.e. the ensemble)
    for conv_kernel in conv_kernels[1:]:
        y = vmap(partial(conv_block, 2, True))(y, conv_kernel)

    # fully convolutional dense layer (with relu) as bottleneck
    y = vmap(partial(conv_block, 1, True))(y, dense_kernels)

    # final projection to embedding dim (with no activation); embeddings are
    # squeezed and unit length normalised.
    embeddings = vmap(partial(conv_block, 1, False))(y, embedding_kernels)
    embeddings = jnp.squeeze(embeddings)  # (M, N, 32)
    embeddings /= jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings  # (M=models, N=num_inputs, E=embedding_dim)
