import tensorflow as tf
from tensorflow.keras import layers as layers


class GAN(tf.keras.Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        super().compile(*args, **kwargs)

        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_step(self, batch):

        # get data
        real_images = batch
        fake_images = self.generator(tf.random.normal((128, 128, 1)), training=False)  # 128 random 128x1 tensors

        # train discriminator
        with tf.GradientTape() as d_tape:

            # pass real and fake images to discriminator model
            yhat_real = self.discriminator(real_images, training=True)
            yhat_fake = self.discriminator(fake_images, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

            # create labels (0: real, 1: fake)
            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)

            # add some noise to the outputs
            noise_real = +0.15 * tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15 * tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)

            # compute loss
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)

        # backpropagation
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        # reshapes random values to 7x7x128
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

        # Convolutioning pt. I
        self.conv1 = layers.Conv2D(32, 5, input_shape=(28, 28, 1))  # (28 x 28 x 1) ---> (24 x 24 x 32)
        self.relu1 = layers.LeakyReLU(0.2)
        self.drop1 = layers.Dropout(0.4)

        # Convolutioning pt. II
        self.conv2 = layers.Conv2D(64, 5)  # (24 x 24 x 32) ---> (20 x 20 x 64)
        self.relu2 = layers.LeakyReLU(0.2)
        self.drop2 = layers.Dropout(0.4)

        # Convolutioning pt. III
        self.conv3 = layers.Conv2D(128, 5)  # (20 x 20 x 64) ---> (16 x 16 x 128)
        self.relu3 = layers.LeakyReLU(0.2)
        self.drop3 = layers.Dropout(0.4)

        # Convolutioning pt. IV
        self.conv4 = layers.Conv2D(256, 5)  # (16 x 16 x 128) ---> (12 x 12 x 256)
        self.relu4 = layers.LeakyReLU(0.2)
        self.drop4 = layers.Dropout(0.4)

        # Flattening
        self.flatt = layers.Flatten()
        self.drop5 = layers.Dropout(0.4)
        self.dense = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        # Convolutioning pt. I
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.drop1(x)

        # Convolutioning pt. II
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        # Convolutioning pt. III
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.drop3(x)

        # Convolutioning pt. IV
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.drop4(x)

        # Flattening
        x = self.flatt(x)
        x = self.drop5(x)
        x = self.dense(x)

        return x
