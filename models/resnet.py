from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, Add, UpSampling2D, LeakyReLU, Lambda
import keras.backend as K
from layers.instance_normalization import InstanceNormalization


class ResNet(object):
    """
    A ResNet neural network for image synthesis
    """
    def __init__(self, input_shape, norm=None, nb_blocks=6, name=''):
        """
        Resnet constructor.
        :param input_shape: image shape (H, W, 1)
        :param norm:        layer normalisation: Can be batch for BatchNormalization or norm for InstanceNormalization
        :param nb_blocks:   number of residual blocks
        :param name:        model name
        """
        self.input_shape = input_shape
        self.norm        = norm
        self.nb_blocks   = nb_blocks
        self.name        = name
        self.model       = None

    def build(self):
        """
        Build the model
        """
        input = Input(self.input_shape)
        l = self.downsample(input)
        l = self.residuals(l)
        l = self.upsample(l)
        self.output(input, l)

    def downsample(self, input):
        """
        Build downsample layers: c7s1-32, c3s2-64, c3s2-128
        :param input:     input layer
        :return:          last layer of downsample operation
        """
        f = 32

        # c7s1-32
        l = Conv2D(f, 7, padding='same')(input)
        l = normalise(self.norm, inshape=K.int_shape(l))(l)
        l = LeakyReLU()(l)

        # c3s2-64
        l = Conv2D(f * 2, 3, strides=2, padding='same')(l)
        l = normalise(self.norm, inshape=K.int_shape(l))(l)
        l = LeakyReLU()(l)

        # c3s2-128
        l = Conv2D(f * 4, 3, strides=2, padding='same')(l)
        l = normalise(self.norm, inshape=K.int_shape(l))(l)
        l = LeakyReLU()(l)

        return l

    def residuals(self, l, f=32 * 4):
        """
        Build residual layers: R128 * nb_blocks
        :param l: input layers
        :param f: number of feature maps
        :return:  the last layer of the residuals
        """
        for block in range(self.nb_blocks):
            l = residual_block(l, f, self.norm)
        return l

    def upsample(self, l):
        """
        Build uplample layers: u64, u32
        :param l: input layer
        :return:  the last layer of the upsample operation
        """
        f = 32

        # u64
        l = UpSampling2D(size=2)(l)
        l = Conv2D(f * 2, 3, padding='same')(l)
        l = normalise(self.norm, inshape=K.int_shape(l))(l)
        l = LeakyReLU()(l)

        # u32
        l = UpSampling2D(size=2)(l)
        l = Conv2D(f, 3, padding='same')(l)
        l = normalise(self.norm, inshape=K.int_shape(l))(l)
        l = LeakyReLU()(l)

        return l

    def output(self, input, l):
        """
        Build last output layer and a ResNet model
        :param input: input layer
        :param l:     last upsample layer
        """
        l = Conv2D(1, 7, activation='tanh', padding='same')(l)
        self.model = Model(inputs=input, outputs=l, name=self.name)


def normalise(norm=None, **kwargs):
    """
    Build a Keras normalization layer
    :param norm: normalization option
    :return:     a normalization layer
    """
    if norm == 'instance':
        return InstanceNormalization(**kwargs)
    elif norm == 'batch':
        return BatchNormalization()
    else:
        return Lambda(lambda x: x)


def residual_block(l0, f, norm):
    """
    Build residual block
    :param l0:      first layer
    :param f:       number of feature maps
    :param norm:    normalization type
    :return:        last layer of the block
    """
    l = Conv2D(f, 3, strides=1, padding='same')(l0)
    l = normalise(norm)(l)
    l = LeakyReLU()(l)
    l = Conv2D(f, 3, strides=1, padding='same')(l)
    l = normalise(norm)(l)
    l = Add()([l0, l])
    return LeakyReLU()(l)
