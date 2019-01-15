from keras import Input, Model
from keras.layers import Conv2D, LeakyReLU, Flatten, Dense
import logging
log = logging.getLogger('discriminator')

class Discriminator(object):
    """
    LS-GAN Discriminator
    """
    def __init__(self, inp_shape, output='2D', downsample_blocks=3, name=''):
        """
        Discriminator constructor
        :param inp_shape:           3-D input shape: (H, W, 1)
        :param output:              can be 1D if there's a single decision or 2D if the output is a 2D image
        :param downsample_blocks:   number of downsample blocks
        :param name:                Model name
        """
        super(Discriminator, self).__init__()

        self.inp_shape         = inp_shape
        self.output            = output
        self.downsample_blocks = downsample_blocks
        self.name              = name
        self.model             = None

    def build(self):
        f = 32

        d_input = Input(self.inp_shape)
        l = Conv2D(f, 4, strides=2, padding='same')(d_input)
        l = LeakyReLU(0.2)(l)

        for i in range(self.downsample_blocks):
            s = 1 if i == self.downsample_blocks - 1 else 2
            l = Conv2D(f * 2 * (2 ** i), 4, strides=s, padding='same')(l)
            l = LeakyReLU(0.2)(l)

        if self.output == '2D':
            l = Conv2D(1, 4, padding='same')(l)
        elif self.output == '1D':
            l = Flatten()(l)
            l = Dense(1, activation='linear')(l)

        self.model = Model(d_input, l, name=self.name)
        log.info('Discriminator %s' % self.name)
        self.model.summary(print_fn=log.info)

