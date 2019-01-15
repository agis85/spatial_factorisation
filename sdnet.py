import logging
import os

import matplotlib

matplotlib.use('Agg')

from keras import Model
from keras.layers import Input, Flatten, Dense, Concatenate, Conv2D, Reshape, BatchNormalization, LeakyReLU, \
    UpSampling2D
from keras.optimizers import Adam

import costs
from loaders import loader_factory
from models.discriminator import Discriminator
from layers.rounding import Rounding
from models.resnet import ResNet
from models.unet import UNet

log = logging.getLogger('sdnet')


class SDNet(object):
    """
    SDNet model for semi-supervised segmentation.
    """

    def __init__(self, conf):
        """
        SDNet constructor
        :param conf: configuration object
        """
        super(SDNet, self).__init__()
        self.other_masks = None
        self.conf = conf
        self.loader = loader_factory.init_loader(self.conf.dataset_name)

        self.D_model            = None  # Discriminator trainer
        self.G_model            = None  # Unsupervised generator trainer
        self.G_supervised_model = None  # Supervised generator trainer
        self.Decomposer         = None  # Decomposer
        self.Reconstructor      = None  # Reconstructor
        self.ImageDiscriminator = None  # Image discriminator
        self.MaskDiscriminator  = None  # Mask discriminator

    def build(self):
        self.build_discriminator_trainer()
        self.build_generator_trainer()
        self.load_models()

    def load_models(self):
        """
        Load weights from saved model files
        """
        if os.path.exists(self.conf.folder + '/D_model'):
            print('Loading trained D_model from file')
            self.D_model.load_weights(self.conf.folder + '/D_model')
            self.ImageDiscriminator = get_net(self.D_model, 'D_X')
            self.MaskDiscriminator  = get_net(self.D_model, 'D_M')

        if os.path.exists(self.conf.folder + '/G_model'):
            print('Loading trained G_model from file')
            self.G_model.load_weights(self.conf.folder + '/G_model')
            self.Decomposer = get_net(self.G_model, 'Decomposer')
            self.Reconstructor = get_net(self.G_model, 'Reconstructor')

        if os.path.exists(self.conf.folder + '/G_supervised_model'):
            print('Loading trained G_supervised_model from file')
            self.G_supervised_model.load_weights(self.conf.folder + '/G_supervised_model')
            self.Decomposer = get_net(self.G_model, 'Decomposer')
            self.Reconstructor = get_net(self.G_model, 'Reconstructor')

    def save_models(self):
        """
        Save model weights in files.
        """
        self.D_model.save_weights(self.conf.folder + '/D_model')
        self.G_model.save_weights(self.conf.folder + '/G_model')
        self.G_supervised_model.save_weights(self.conf.folder + '/G_supervised_model')

    def build_discriminator_trainer(self):
        """
        Build a Keras model for training image and mask discriminators.
        """
        # Mask Discriminator
        D_Mask = Discriminator(self.conf.input_shape, output='2D', downsample_blocks=3, name='D_M')
        D_Mask.build()
        self.MaskDiscriminator = D_Mask.model

        real_M = Input(self.conf.input_shape)
        fake_M = Input(self.conf.input_shape)
        dis_real_M = self.MaskDiscriminator(real_M)
        dis_fake_M = self.MaskDiscriminator(fake_M)

        D_Image = Discriminator(self.conf.input_shape, output='2D', downsample_blocks=3, name='D_X')
        D_Image.build()
        self.ImageDiscriminator = D_Image.model

        real_X = Input(self.conf.input_shape)
        fake_X = Input(self.conf.input_shape)
        dis_real_X = self.ImageDiscriminator(real_X)
        dis_fake_X = self.ImageDiscriminator(fake_X)

        self.D_model = Model(inputs=[real_M, fake_M, real_X, fake_X],
                             outputs=[dis_real_M, dis_fake_M, dis_real_X, dis_fake_X])
        self.D_model.compile(Adam(lr=0.0001, beta_1=0.5), loss='mse')
        log.info('Discriminators Trainer')
        self.D_model.summary(print_fn=log.info)

    def build_generator_trainer(self):
        """
        Build Decomposer, Reconstructor and training models.
        """
        assert self.D_model is not None, 'Discriminator has not been built yet'
        make_trainable(self.D_model, False)

        self.Decomposer = self._decomposer()
        self.Reconstructor = self._reconstructor()

        self.build_unsupervised_trainer()
        self.build_supervised_trainer()

    def build_unsupervised_trainer(self):
        """
        Build a Keras model for training SDNet with no supervision, using adversarial training with a mask
        discriminator and an image reconstruction cost.
        """
        # Decomposition/Segmentation X -> M', Z
        real_X = Input(self.conf.input_shape)
        fake_M, fake_Z = self.Decomposer(real_X)
        adv_M = self.MaskDiscriminator(fake_M)

        # Reconstruction M', Z' -> X'
        rec_X = self.Reconstructor([fake_M, fake_Z])
        adv_X = self.ImageDiscriminator(rec_X)

        self.G_model = Model(inputs=real_X, outputs=[adv_M, rec_X, adv_X])
        self.G_model.compile(Adam(lr=0.0001, beta_1=0.5), loss=['mse', 'mae', 'mse'],
                             loss_weights=[self.conf.w_uns_adv_M, self.conf.w_uns_rec_X, self.conf.w_uns_adv_X])
        log.info('Unsupervised trainer')
        self.G_model.summary(print_fn=log.info)

    def build_supervised_trainer(self):
        """
        Build a Keras model for training SDNet with supervision, when we have labelled data.
        """
        # Decomposition/Segmentation X -> M', Z'
        real_X = Input(self.conf.input_shape)
        fake_M, fake_Z = self.Decomposer(real_X)
        adv_M = self.MaskDiscriminator(fake_M)

        # Reconstruction M', Z' -> X'
        rec_X = self.Reconstructor([fake_M, fake_Z])

        # Reconstruction using a real Mask: M, Z' -> X'
        real_M = Input(self.conf.input_shape)
        fake_X = self.Reconstructor([real_M, fake_Z])
        adv_X = self.ImageDiscriminator(fake_X)

        self.G_supervised_model = Model(inputs=[real_X, real_M], outputs=[fake_M, fake_X, rec_X, adv_M, adv_X])
        self.G_supervised_model.compile(Adam(lr=0.0001, beta_1=0.5),
                                        loss=[costs.dice_coef_loss, 'mae', 'mae', 'mse', 'mse'],
                                        loss_weights=[self.conf.w_fake_M, self.conf.w_fake_X, self.conf.w_rec_X,
                                                      self.conf.w_adv_M, self.conf.w_adv_X_fromreal])
        log.info('Supervised trainer')
        self.G_supervised_model.summary(print_fn=log.info)

    def _decomposer(self):
        """
        Build an image decomposer into a spatial binary mask of the myocardium and a non-spatial vector z of the
        remaining image information.
        :return a Keras model of the decomposer
        """
        input = Input(self.conf.input_shape)

        unet = UNet(self.conf.input_shape, residual=False)
        l = unet.unet_downsample(input)
        unet.unet_bottleneck(l)
        l = unet.bottleneck

        # build Z regressor
        modality = Conv2D(256, 3, strides=1, padding='same')(l)
        modality = BatchNormalization()(modality)
        modality = LeakyReLU()(modality)
        modality = Conv2D(64, 3, strides=1, padding='same')(modality)
        modality = BatchNormalization()(modality)
        modality = LeakyReLU()(modality)
        modality = Flatten()(modality)
        modality = Dense(32)(modality)
        modality = LeakyReLU()(modality)
        modality = Dense(16, activation='sigmoid')(modality)

        l = unet.unet_upsample(unet.bottleneck)
        anatomy = unet.out(l)

        m = Model(inputs=input, outputs=[anatomy, modality], name='Decomposer')
        log.info('Decomposer')
        m.summary(print_fn=log.info)
        return m

    def _reconstructor(self):
        """
        Build an image reconstructor, that fuses an anatomy (binary mask) and Z to reconstructs the input image.
        :return: a Keras model of the reconstructor
        """
        mask_input = Input(shape=self.conf.input_shape)
        round = Rounding()(mask_input)  # rounding layer that binarises the anatomical representation.

        resnet = ResNet(self.conf.input_shape, norm='instance', nb_blocks=3, name='Reconstructor')

        # Map Z into a 8-channel feature map
        resd_input = Input((16,))
        modality = Dense(32)(resd_input)
        modality = LeakyReLU()(modality)
        modality = Dense(self.conf.input_shape[0] * self.conf.input_shape[1])(modality)
        modality = LeakyReLU()(modality)
        modality = Reshape((int(self.conf.input_shape[0] / 4), int(self.conf.input_shape[1] / 4), 16))(modality)
        modality = UpSampling2D(size=2)(modality)
        modality = Conv2D(16, 3, padding='same')(modality)
        modality = BatchNormalization()(modality)
        modality = LeakyReLU()(modality)
        modality = UpSampling2D(size=2)(modality)
        modality = Conv2D(8, 3, padding='same')(modality)
        modality = BatchNormalization()(modality)
        modality = LeakyReLU()(modality)

        # Concatenate Mask and Z
        conc_lr = Concatenate()([round, modality])
        l = resnet.residuals(conc_lr, f=9)
        resnet.output([mask_input, resd_input], l)
        resnet.model.summary(print_fn=log.info)
        return resnet.model


def get_net(trainer_model, name):
    """
    Helper method to get a layer with a given name out of a model
    :param trainer_model: base model
    :param name:          layer name
    :return:              a layer with the specified name
    """
    layers = [l for l in trainer_model.layers if l.name == name]
    assert len(layers) == 1
    return layers[0]


def make_trainable(model, val):
    """
    Helper method to enable/disable training of a model
    :param model: a Keras model
    :param val:   True/False
    """
    model.trainable = val
    try:
        for l in model.layers:
            try:
                for k in l.layers:
                    make_trainable(k, val)
            except:
                # Layer is not a model, so continue
                pass
            l.trainable = val
    except:
        # Layer is not a model, so continue
        pass
