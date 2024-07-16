import tensorflow as tf

IMG_HEIGHT = 50
IMG_WIDTH = 50


class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.data_aug1 = tf.keras.layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        self.data_aug2 = tf.keras.layers.RandomRotation(0.1),
        self.data_aug3 = tf.keras.layers.RandomZoom(0.1),
        self.scale = tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))  # [0, 255] ---> [0, 1]
        self.conv1 = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D()
        self.conv3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D()
        self.dropout = tf.keras.layers.Dropout(0.2),
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        # x = self.data_aug1(inputs)
        # x = self.data_aug2(x)
        # x = self.data_aug3(x)
        x = self.scale(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        # x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)
