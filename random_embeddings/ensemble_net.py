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


class EnsembleNet(object):

    def __init__(self, num_models):
        self.num_models = num_models
        key = random.PRNGKey(0)
        subkeys = random.split(key, 8)
        self.conv1_kernels = orthogonal()(
            subkeys[0], (num_models, 3, 3, 3, 32))
        self.conv2_kernels = orthogonal()(
            subkeys[1], (num_models, 3, 3, 32, 64))
        self.conv3_kernels = orthogonal()(
            subkeys[2], (num_models, 3, 3, 64, 128))
        self.conv4_kernels = orthogonal()(
            subkeys[3], (num_models, 3, 3, 128, 128))
        self.conv5_kernels = orthogonal()(
            subkeys[4], (num_models, 3, 3, 128, 128))
        self.conv6_kernels = orthogonal()(
            subkeys[5], (num_models, 3, 3, 128, 128))
        self.dense_kernels = orthogonal()(
            subkeys[6], (num_models, 1, 1, 128, 32))
        self.embedding_kernels = orthogonal()(
            subkeys[7], (num_models, 1, 1, 32, 32))

    def embed(self, input):
        # TODO decorating this method directly fails?
        return jit(self._embed)(input)

    def _embed(self, input):

        # convolutional stack; stride 2 for downsizing
        y = vmap(partial(conv_block, 2, True, input))(self.conv1_kernels)
        y = vmap(partial(conv_block, 2, True))(y, self.conv2_kernels)
        y = vmap(partial(conv_block, 2, True))(y, self.conv3_kernels)
        y = vmap(partial(conv_block, 2, True))(y, self.conv4_kernels)
        y = vmap(partial(conv_block, 2, True))(y, self.conv5_kernels)
        y = vmap(partial(conv_block, 2, True))(y, self.conv5_kernels)

        # fully convolutional dense layer to project down
        y = vmap(partial(conv_block, 1, True))(y, self.dense_kernels)

        # final projection to embedding dim (with no activation), squeezed and
        # unit length normalised.
        embeddings = vmap(partial(conv_block, 1, False))(
            y, self.embedding_kernels)
        embeddings = jnp.squeeze(embeddings)  # (M, N, 32)
        embeddings /= jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
        return embeddings
