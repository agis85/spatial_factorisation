from skimage.measure import block_reduce
import numpy as np
import logging
log = logging.getLogger('data')


class Data(object):
    """
    Object that stores images and masks loaded from a dataset. It defines useful functions for data manipulation and
    data selection.
    """

    def __init__(self, images, masks, index, downsample=1):
        """
        Data constructor.
        :param images:      a 4-D numpy array of images. Expected shape: (N, H, W, 1)
        :param masks:       a 4-D numpy array of myocardium segmentation masks. Expected shape: (N, H, W, 1)
        :param index:       a 1-D numpy array indicating the volume each image/mask belongs to. Used for data selection.
        """
        if images is None:
            raise ValueError('Images cannot be None.')
        if masks is None:
            raise ValueError('Masks cannot be None.')
        if index is None:
            raise ValueError('Index cannot be None.')
        if images.shape != masks.shape:
            raise ValueError('Image shape=%s different from Mask shape=%s' % (str(images.shape), str(masks.shape)))
        if images.shape[0] != index.shape[0]:
            raise ValueError('Different number of images and indices: %d vs %d' % (images.shape[0], index.shape[0]))

        self.images = images
        self.masks  = masks
        self.index  = index
        self.downsample(downsample)
        num_volumes = len(self.volumes())
        log.info('Created Data object with images of shape %s and %d volumes' % (str(images.shape), num_volumes))
        log.info('Images value range [%.1f, %.1f]' % (images.min(), images.max()))
        log.info('Masks value range [%.1f, %.1f]' % (masks.min(), masks.max()))

    def volumes(self):
        return sorted(set(self.index))

    def get_volume_image(self, vol):
        return self.images[self.index == vol]

    def get_volume_mask(self, vol):
        return self.masks[self.index == vol]

    def size(self):
        return len(self.images)

    def resize(self, num):
        self.images = self.images[:num]
        self.masks = self.masks[:num]

    def shape(self):
        return self.images.shape

    def downsample(self, ratio=2):
        if ratio == 1:
            return

        self.images = block_reduce(self.images, block_size=(1, ratio, ratio, 1), func=np.mean)
        if self.masks is not None:
            self.masks = block_reduce(self.masks, block_size=(1, ratio, ratio, 1), func=np.mean)
        log.info('Downsampled data by %d to shape %s' % (ratio, str(self.images.shape)))

    def sample(self, nb_samples, seed=-1):
        log.info('Sampling %d images out of total %d' % (nb_samples, self.size()))
        if seed > -1:
            np.random.seed(seed)

        idx = np.random.choice(self.size(), size=nb_samples, replace=False)
        log.debug('Indices sampled: ' + str(idx))
        self.images = np.array([self.images[i] for i in idx])
        self.masks  = np.array([self.masks[i] for i in idx])
        self.index  = np.array([self.index[i] for i in idx])
