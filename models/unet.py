from keras import Input, Model
from keras.layers import Concatenate, Conv2D, BatchNormalization, LeakyReLU, UpSampling2D, Activation, MaxPooling2D


class UNet(object):
    """
    UNet implementation of 4 downsampling and 4 upsampling blocks for segmentation.
    Each block has 2 convolutions, batch normalisation, leaky relu and an optional residual connection.
    The number of filters for the 1st layer is 64 and at every block, this is doubled. Each upsampling blocks halves the
    number of filters.
    """
    def __init__(self, input_shape, residual, f=64):
        """
        Constructor
        :param input_shape:         image shape: (H, W, 1)
        :param residual:            option for residual blocks in the downsampling and upsampling paths
        :param f:                   number of feature maps in the first layer
        """
        self.input_shape  = input_shape
        self.residual     = residual
        self.f            = f

        # model layers
        self.model      = None # the Keras model
        self.input      = None # input layer
        self.d_l0       = None # downsample layer 1
        self.d_l1       = None # downsample layer 2
        self.d_l2       = None # downsample layer 3
        self.d_l3       = None # downsample layer 4
        self.bottleneck = None # most downsampled UNet layer
        self.u_l3       = None # upsample layer 1
        self.u_l2       = None # upsample layer 2
        self.u_l1       = None # upsample layer 3
        self.u_l0       = None # upsample layer 4

    def build(self):
        """
        Build the model.
        """
        self.input = Input(shape=self.input_shape)
        l = self.unet_downsample(self.input)
        self.unet_bottleneck(l)
        l = self.unet_upsample(self.bottleneck)
        out = self.out(l)
        self.model = Model(inputs=self.input, outputs=out)

    def unet_downsample(self, inp):
        """
        Build downsampling path
        :param inp: input layer
        :return:    last layer of the downsampling path
        """
        self.d_l0 = conv_block(inp, self.f, self.residual)
        l = MaxPooling2D(pool_size=(2, 2))(self.d_l0)
        self.d_l1 = conv_block(l, self.f * 2, self.residual)
        l = MaxPooling2D(pool_size=(2, 2))(self.d_l1)
        self.d_l2 = conv_block(l, self.f * 4, self.residual)
        l = MaxPooling2D(pool_size=(2, 2))(self.d_l2)
        self.d_l3 = conv_block(l, self.f * 8)
        l = MaxPooling2D(pool_size=(2, 2))(self.d_l3)
        return l

    def unet_bottleneck(self, l):
        """
        Build bottleneck layers
        :param l: the input layer
        """
        self.bottleneck = conv_block(l, self.f * 16, self.residual)

    def unet_upsample(self, l):
        """
        Build upsampling path
        :param l: the input layer
        :return:  the last layer of the upsampling path
        """
        l = upsample_block(l, self.f * 8, activation='linear')
        l = Concatenate()([l, self.d_l3])
        self.u_l3 = conv_block(l, self.f * 8)
        l = upsample_block(self.u_l3, self.f * 4, activation='linear')
        l = Concatenate()([l, self.d_l2])
        self.u_l2 = conv_block(l, self.f * 4, self.residual)
        l = upsample_block(self.u_l2, self.f * 2, activation='linear')
        l = Concatenate()([l, self.d_l1])
        self.u_l1 = conv_block(l, self.f * 2, self.residual)
        l = upsample_block(self.u_l1, self.f, activation='linear')
        l = Concatenate()([l, self.d_l0])
        self.u_l0 = conv_block(l, self.f, self.residual)
        return self.u_l0

    def out(self, l):
        """
        Build ouput layer
        :param l: last layer from the upsampling path
        :return:  the final segmentation layer
        """
        return Conv2D(1, 1, activation='sigmoid')(l)


def conv_block(l0, f, residual=False):
    """
    Convolutional block of the downsampling path
    :param l0:        the input layer
    :param f:         number of feature maps
    :param residual:  True/False to define residual connections
    :return:          the last layer of the convolutional block
    """
    l = Conv2D(f, 3, strides=1, padding='same')(l0)
    l = BatchNormalization()(l)
    l = LeakyReLU()(l)
    l = Conv2D(f, 3, strides=1, padding='same')(l)
    l = BatchNormalization()(l)
    l = LeakyReLU()(l)
    return Concatenate()([l0, l]) if residual else l


def upsample_block(l0, f, activation='relu'):
    """
    Upsampling block.
    :param l0:          input layer
    :param f:           number of feature maps
    :param activation:  activation name
    :return:            the last layer of the upsampling block
    """
    l = UpSampling2D(size=2)(l0)
    l = Conv2D(f, 3, padding='same')(l)
    l = BatchNormalization()(l)
    return Activation(activation)(l)
