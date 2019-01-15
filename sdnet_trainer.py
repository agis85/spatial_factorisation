import logging
import os
import numpy as np
import scipy
from keras.callbacks import CSVLogger, EarlyStopping
from keras.utils import Progbar

import costs
from callbacks.loss_callback import SaveLoss
from callbacks.sdnet_callback import SDNetCallback
from loaders import loader_factory
from utils import data_utils

log = logging.getLogger('sdnettrainer')


class SDNetTrainer(object):
    """
    Trainer class for running a segmentation experiment using SDNet.
    """
    def __init__(self, sdnet, conf):
        self.sdnet   = sdnet
        self.conf    = conf
        self.loader  = loader_factory.init_loader(self.conf.dataset_name)

        # Data iterators
        self.gen_X_L     = None # labelled data: (image, mask) pairs
        self.gen_X_U     = None # unlabelled data
        self.other_masks = None # real masks to use for discriminator training

        self.fake_image_pool = []
        self.fake_mask_pool  = []
        self.batch = 0
        self.epoch = 0

        if not os.path.exists(self.conf.folder):
            os.makedirs(self.conf.folder)

    def init_train(self):
        """
        Initialise data generators for iterating through the images and masks.
        """
        data = self.loader.load_labelled_data(self.conf.split, 'training')

        # Initialise unlabelled data iterator
        num_ul = 0
        if self.conf.ul_mix > 0:
            ul_data = self.loader.load_unlabelled_data(self.conf.split, 'all')

            # calculate number of unlabelled images as a proportion of the labelled images
            num_ul = int(data.size() * self.conf.ul_mix)
            num_ul = num_ul if num_ul <= ul_data.size() else ul_data.size()
            log.info('Sampling %d unlabelled images out of total %d.' % (num_ul, ul_data.size()))
            ul_data.sample(num_ul)
            self.gen_X_U = data_utils.generator(self.conf.batch_size, 'overflow', ul_data.images)

        # Initialise labelled data iterator
        assert self.conf.l_mix >= 0

        # calculate number of labelled images
        num_l = int(data.size() * self.conf.l_mix)
        num_l = num_l if num_l <= data.size() else data.size()
        log.info('Using %d labelled images out of total %d.' % (num_l, data.size()))
        train_images = data.images[:num_l]
        train_masks = data.masks[:num_l]

        self.conf.unlabelled_image_num = num_ul
        self.conf.labelled_image_num = num_l
        self.conf.data_len = num_ul if num_ul > num_l else num_l
        self.conf.batches = int(np.ceil(self.conf.data_len / self.conf.batch_size))
        self.conf.save()

        self.gen_X_L = data_utils.generator(self.conf.batch_size, 'overflow', train_images, train_masks)

        # Initialise real masks iterator for discriminator training, using the real masks from the data CV split.
        self.other_masks = data_utils.generator(self.conf.batch_size, 'overflow', data.masks + 0)

    def fit(self):
        """
        Train SDNet
        """
        log.info('Training SDNet')

        # Load data
        self.init_train()

        # Initialise callbacks
        sl = SaveLoss(self.conf.folder)
        cl = CSVLogger(self.conf.folder + '/training.csv')
        cl.on_train_begin()
        si = SDNetCallback(self.conf.folder, self.conf.batch_size, self.sdnet)
        es = EarlyStopping('val_loss', min_delta=0.001, patience=20)
        es.on_train_begin()

        loss_names = ['adv_M', 'adv_X', 'rec_X', 'rec_M', 'rec_Z', 'dis_M', 'dis_X', 'mask', 'image', 'val_loss']

        total_loss = {n: [] for n in loss_names}

        progress_bar = Progbar(target=self.conf.batches * self.conf.batch_size)

        for self.epoch in range(self.conf.epochs):
            log.info('Epoch %d/%d' % (self.epoch, self.conf.epochs))

            real_lb_pool, real_ul_pool = [], []  # these are used only for printing images

            epoch_loss = {n: [] for n in loss_names}

            D_initial_weights = np.mean([np.mean(w) for w in self.sdnet.D_model.get_weights()])
            G_initial_weights = np.mean([np.mean(w) for w in self.sdnet.G_model.get_weights()])
            for self.batch in range(self.conf.batches):
                real_lb = next(self.gen_X_L)
                real_ul = next(self.gen_X_U)

                # Add image/mask batch to the data pool
                x, m = real_lb
                real_lb_pool.extend([(x[i:i+1], m[i:i+1]) for i in range(x.shape[0])])
                real_ul_pool.extend(real_ul)

                D_weights1 = np.mean([np.mean(w) for w in self.sdnet.D_model.get_weights()])
                self.train_batch_generator(real_lb, real_ul, epoch_loss)
                D_weights2 = np.mean([np.mean(w) for w in self.sdnet.D_model.get_weights()])
                assert D_weights1 == D_weights2

                self.train_batch_discriminator(real_lb, real_ul, epoch_loss)

                progress_bar.update((self.batch + 1) * self.conf.batch_size)

            G_final_weights = np.mean([np.mean(w) for w in self.sdnet.G_model.get_weights()])
            D_final_weights = np.mean([np.mean(w) for w in self.sdnet.D_model.get_weights()])

            # Check training is altering weights
            assert D_initial_weights != D_final_weights
            assert G_initial_weights != G_final_weights

            # Plot some example images
            si.on_epoch_end(self.epoch, np.array(real_lb_pool), np.array(real_ul_pool))

            self.validate(epoch_loss)

            # Calculate epoch losses
            for n in loss_names:
                total_loss[n].append(np.mean(epoch_loss[n]))
            log.info(str('Epoch %d/%d: ' + ', '.join([l + ' Loss = %.3f' for l in loss_names])) % \
                  ((self.epoch, self.conf.epochs) + tuple(total_loss[l][-1] for l in loss_names)))
            logs = {l: total_loss[l][-1] for l in loss_names}
            sl.on_epoch_end(self.epoch, logs)

            # log losses to csv
            cl.model = self.sdnet.D_model
            cl.model.stop_training = False
            cl.on_epoch_end(self.epoch, logs)

            # save models
            self.sdnet.save_models()

            # early stopping
            if self.stop_criterion(es, self.epoch, logs):
                log.info('Finished training from early stopping criterion')
                break

    def train_batch_generator(self, real_lb, real_ul, epoch_loss):
        """
        Train Generator networks. This is done in two passes:
        (a) Unsupervised training using unlabelled data and masks
        (b) Supervised training using labelled data and masks.
        :param real_lb:    labelled tuple of images and masks
        :param real_ul:    unlabelled images
        :param epoch_loss: loss dictionary for the epoch
        """

        X_L, X_M = real_lb
        X_U      = real_ul

        # Train unlabelled path (G_model)
        adv_M, rec_X, adv_X = [], [], []
        if X_U.shape[0] > 0:
            zeros = np.zeros((X_U.shape[0],) + self.sdnet.D_model.output_shape[0][1:])
            _, ul_l_adv_M, ul_l_rec_X, ul_l_adv_X = self.sdnet.G_model.train_on_batch(X_U, [zeros, X_U, zeros])
            assert np.mean(ul_l_adv_M) >= 0, "loss_fake_M: " + str(ul_l_adv_M)
            assert ul_l_rec_X >= 0, "loss_rec_X: " + str(ul_l_rec_X)
            adv_M.append(ul_l_adv_M)
            rec_X.append(ul_l_rec_X)
            adv_X.append(ul_l_adv_X)

        # Train labelled path (G_supervised_model)
        if X_L.shape[0] > 0:
            zeros = np.zeros((X_L.shape[0],) + self.sdnet.D_model.output_shape[0][1:])
            _, Z = self.sdnet.Decomposer.predict(X_L)
            x = [X_L, X_M]
            y = [X_M, X_L, X_L, zeros, zeros]
            _, l_mask, l_img, l_rec_X , l_adv_M, l_adv_X = self.sdnet.G_supervised_model.train_on_batch(x, y)
            epoch_loss['mask'].append(l_mask)
            epoch_loss['image'].append(l_img)
            adv_M.append(l_adv_M)
            rec_X.append(l_rec_X)
            adv_X.append(l_adv_X)

        epoch_loss['adv_M'].append(np.mean(adv_M))
        epoch_loss['adv_X'].append(np.mean(adv_X))
        epoch_loss['rec_X'].append(np.mean(rec_X))

    def train_batch_discriminator(self, real_lb, real_ul, epoch_loss):
        """
        Train a discriminator with real X / fake X and real M / fake M. To produce a fake X we use a real M and a Z
        produced by the mask's decomposition
        :param real_lb:     tuple of labelled images and masks
        :param real_B:      unlabelled images
        :param epoch_loss:  dictionary of losses for the epoch
        """
        X_L, X_M = real_lb
        X_U      = real_ul

        # When reaching the end of the array, the array size might be less than the true batch size
        batch_size = np.min([X_L.shape[0], X_U.shape[0]])

        if batch_size < X_M.shape[0]:
            idx = np.random.choice(X_M.shape[0], size=batch_size, replace=False)
            X_M = np.array([X_M[i] for i in idx])

        X = self.sample_X(X_L, X_U, size=batch_size)
        fake_M, Z = self.sdnet.Decomposer.predict(X)
        fake_X    = self.sdnet.Reconstructor.predict([X_M, Z])

        # Pool of fake images. Using one pool regularises the Mask discriminator in the first epochs.
        self.fake_mask_pool,  fake_M = self.get_fake(fake_M, self.fake_image_pool, size=batch_size)
        self.fake_image_pool, fake_X = self.get_fake(fake_X, self.fake_image_pool, size=batch_size)

        # If we have a pool of other images use some of it for real examples
        if self.other_masks is not None:
            M_other = next(self.other_masks)
            X_M = data_utils.sample(np.concatenate([X_M, M_other], axis=0), batch_size)

        # Train Discriminator
        zeros = np.zeros((X_M.shape[0],) + self.sdnet.D_model.output_shape[0][1:])
        ones  = np.ones(zeros.shape)

        x = [X_M, fake_M, X, fake_X]
        y = [zeros, ones, zeros, ones]
        _, D_loss_real_M, D_loss_fake_M, D_loss_real_X, D_loss_fake_X = self.sdnet.D_model.train_on_batch(x, y)
        epoch_loss['dis_M'].append(np.mean([D_loss_real_M, D_loss_fake_M]))
        epoch_loss['dis_X'].append(np.mean([D_loss_real_X, D_loss_fake_X]))

    def sample_X(self, X_L, X_U, size):
        """
        Sample images of size=batch size from the labelled and unlabelled array. If we've passed through all
        labelled images, ignore them when sampling (and vice versa).
        :param X_L:           array of labelled images
        :param X_U:           array of unlabelled images
        :return:              an image array to be used for calculating fake_masks
        """
        # find the batch number that the iterator of the labelled (or unlabelled) images finishes
        bn_end = np.min([self.conf.unlabelled_image_num, self.conf.labelled_image_num]) / self.conf.batch_size
        if self.batch < bn_end:
            all = np.concatenate([X_L, X_U], axis=0)
            idx = np.random.choice(all.shape[0], size=size, replace=False)
            X = np.array([all[i] for i in idx])
        elif self.conf.labelled_image_num > self.conf.unlabelled_image_num:
            idx = np.random.choice(X_L.shape[0], size=size, replace=False)
            X = np.array([X_L[i] for i in idx])
        else:
            idx = np.random.choice(X_U.shape[0], size=size, replace=False)
            X = np.array([X_U[i] for i in idx])

        return X

    def get_fake(self, pred, fake_pool, size):
        """
        Add item to the pool of data. Then select a random number of items from the pool.
        :param pred:        new datum to add to the pool
        :param fake_pool:   the data pool of fake images/masks
        :return:            the sampled data from the pool
        """
        fake_pool.extend(pred)
        fake_pool = fake_pool[-self.conf.pool_size:]
        sel = np.random.choice(len(fake_pool), size=size, replace=False)
        fake_A = np.array([fake_pool[ind] for ind in sel])
        return fake_pool, fake_A

    def validate(self, epoch_loss):
        """
        Report validation error
        :param epoch_loss: dictionary of losses
        """
        valid_data = self.loader.load_labelled_data(self.conf.split, 'validation')
        mask, _ = self.sdnet.Decomposer.predict(valid_data.images)
        assert mask.shape == valid_data.masks.shape
        epoch_loss['val_loss'].append((1-costs.dice(valid_data.masks, mask)))

    def stop_criterion(self, es, epoch, logs):
        """
        Criterion for early stopping of training
        :param es:      Keras EarlyStopping callback
        :param epoch:   epoch number
        :param logs:    dictionary of losses
        :return:        True/False: stop training or not
        """
        es.model = self.sdnet.Decomposer
        es.on_epoch_end(epoch, logs)
        if es.stopped_epoch > 0:
            return True

    def test(self):
        """
        Evaluate a model on the test data.
        """
        log.info('Evaluating model on test data')
        folder = os.path.join(self.conf.folder, 'test_results_%s' % self.conf.dataset_name)
        if not os.path.exists(folder):
            os.makedirs(folder)

        test_loader = loader_factory.init_loader(self.conf.dataset_name)
        test_data = test_loader.load_labelled_data(self.conf.split, 'test')

        synth = []
        im_dice = {}
        samples = os.path.join(folder, 'samples')
        if not os.path.exists(samples):
            os.makedirs(samples)

        f = open(os.path.join(folder, 'results.csv'), 'w')
        f.writelines('Vol, Dice\n')

        for vol_i in test_data.volumes():
            vol_folder = os.path.join(samples, 'vol_%s' % str(vol_i))
            if not os.path.exists(vol_folder):
                os.makedirs(vol_folder)

            vol_image = test_data.get_volume_image(vol_i)
            vol_mask = test_data.get_volume_mask(vol_i)
            assert vol_image.shape[0] > 0 and vol_image.shape == vol_mask.shape
            pred, _ = self.sdnet.Decomposer.predict(vol_image)

            synth.append(pred)
            im_dice[vol_i] = costs.dice(vol_mask, pred)
            f.writelines('%s, %.3f\n' % (str(vol_i), im_dice[vol_i]))

            for i in range(vol_image.shape[0]):
                im = np.concatenate([vol_image[i, :, :, 0], pred[i, :, :, 0], vol_mask[i, :, :, 0]], axis=1)
                scipy.misc.imsave(os.path.join(vol_folder, 'test_vol%d_sl%d.png' % (vol_i, i)), im)

        print('Dice score: %.3f' % np.mean(list(im_dice.values())))
        f.close()
