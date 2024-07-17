import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img


def save_fig(name):
    output_dir = '/Users/kerembuekrue/Documents/code/DeepImageception/output/'
    if len(name) > 1:
        output_dir += name[0]
        filename = name[1]
    else:
        filename = name[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, filename + ".png"), dpi=100)


class ModelMonitor(Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.uniform((self.num_img, self.latent_dim, 1))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            img.save(os.path.join('../output/images', f'generated_img_{epoch}_{i}.png'))
