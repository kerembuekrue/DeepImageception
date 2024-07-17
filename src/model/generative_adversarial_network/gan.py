import tensorflow as tf
from tensorflow.keras import layers as layers

IMG_HEIGHT = 28
IMG_WIDTH = 28


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        # reshapes random values to 7x7x128
        # self.dense = layers.Dense(7*7*128, input_dim=128)
        self.dense = layers.Dense(7 * 7 * 128)

        self.relu1 = layers.LeakyReLU(0.2)
        self.reshp = layers.Reshape((7, 7, 128))

        # Upsampling pt. I
        self.upsm1 = layers.UpSampling2D()
        self.conv1 = layers.Conv2D(128, 5, padding="same")
        self.relu2 = layers.LeakyReLU(0.2)

        # Upsampling pt. II
        self.upsm2 = layers.UpSampling2D()
        self.conv2 = layers.Conv2D(128, 5, padding="same")
        self.relu3 = layers.LeakyReLU(0.2)
        #
        # Convolutioning pt. I
        self.conv3 = layers.Conv2D(128, 4, padding="same")
        self.relu4 = layers.LeakyReLU(0.2)

        # Convolutioning pt. II
        self.conv4 = layers.Conv2D(128, 4, padding="same")
        self.relu5 = layers.LeakyReLU(0.2)

        # Get one channel
        self.conv5 = layers.Conv2D(1, 4, padding="same", activation="sigmoid")

    def call(self, inputs):

        x = self.dense(inputs)
        x = self.relu1(x)
        x = self.reshp(x)

        x = self.upsm1(x)
        x = self.conv1(x)
        x = self.relu2(x)

        x = self.upsm2(x)
        x = self.conv2(x)
        x = self.relu3(x)

        x = self.conv3(x)
        x = self.relu4(x)

        x = self.conv4(x)
        x = self.relu5(x)

        x = self.conv5(x)

        return x


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.data_aug1 = tf.keras.layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    def call(self, inputs):

        return 0
