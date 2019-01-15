import json
import os


class Configuration(object):
    """
    Configuration object with experiment parameters.
    """
    def __init__(self, folder, data_len, input_shape):
        self.seed = 0
        self.folder = folder
        self.data_len = data_len
        self.epochs = 500
        self.batch_size = 4
        self.batches = data_len / self.batch_size
        self.pool_size = 50
        self.input_shape = input_shape
        self.split = None
        self.description = ''
        self.dataset_name = ''
        self.unlabelled_image_num = 0
        self.labelled_image_num = 0
        self.augment = False
        self.l_mix = None
        self.ul_mix = None

        self.w_uns_adv_M = 10
        self.w_uns_rec_X = 5
        self.w_uns_adv_X = 5
        self.w_fake_M = 10
        self.w_fake_X = 10
        self.w_rec_X = 1
        self.w_adv_M = 10
        self.w_adv_X_fromreal = 1

    def save(self):
        fname = os.path.join(self.folder, 'config.json')
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        with open(fname, 'w') as outfile:
            json.dump(self.__dict__, outfile)
