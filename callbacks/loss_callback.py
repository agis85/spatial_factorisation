
import os
import matplotlib.pyplot as plt
from keras.callbacks import Callback


class SaveLoss(Callback):
    """
    Keras callback for saving a graph of the training losses over epochs as a .png image.
    It creates two plots, one for the adversarial losses (training_discr_loss.png) and one
    for the remaining losses (training_loss.png).
    """
    def __init__(self, folder, scale='linear'):
        """
        Callback constructor
        :param folder:  folder to save the images
        :param scale:   can be 'linear' or 'log', to plot values in y-axis in a linear or logarithmic scale.
        """
        super(SaveLoss, self).__init__()
        self.folder = folder
        self.values = dict()

        if scale not in ['linear', 'log']:
            raise ValueError('Invalid value for scale. Allowed values: linear, log. Given value: %s' % str(scale))
        self.scale = scale

    # Overwrite default on_epoch_end implementation
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            raise ValueError('Parameter logs cannot be None.')

        # Initialise self.values dictionary the first epoch.
        if len(self.values) == 0:
            for k in logs:
                self.values[k] = []

        # Update dictionary values.
        for k in logs:
            self.values[k].append(logs[k])

        # Save a graph of the training loss values.
        plt.figure()
        plt.suptitle('Training loss', fontsize=16)
        for k in self.values:
            epochs = range(len(self.values[k]))
            if self.scale == 'linear':
                plt.plot(epochs, self.values[k], label=k)
            elif self.scale == 'log':
                plt.semilogy(epochs, self.values[k], label=k)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.folder, 'training_loss.png'))

        # Save a graph of the loss values of adversarial training.
        # Convention: Adversarial loss names start with dis or adv
        plt.figure()
        plt.suptitle('Training loss', fontsize=16)
        for k in self.values:
            if not ('dis' in k or 'adv' in k):
                continue

            epochs = range(len(self.values[k]))
            plt.plot(epochs, self.values[k], label=k)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.folder, 'training_discr_loss.png'))

        plt.close()
