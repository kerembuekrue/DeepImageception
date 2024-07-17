import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from src.model.generative_adversarial_network.gan import Generator, Discriminator
from src.utils.utils import save_fig

# --- LOAD DATA ---------------------------------
ds = tfds.load("fashion_mnist", split="train")

# --- VISUALIZE DATA ----------------------------
datait = ds.as_numpy_iterator()
fig, axs = plt.subplots(3, 5, figsize=(7, 5))
for i in range(3):
    for j in range(5):
        batch = datait.next()
        axs[i, j].imshow(batch["image"])
        axs[i, j].set_title(batch["label"])
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
plt.tight_layout()
save_fig(["gan_training_samples"])

# --- DATA PIPELINE -----------------------------
def scale_images(data):
    image = data["image"]
    return image / 255


ds = ds.map(scale_images)  # Apply the function "scale_images" to every sample of the dataset "ds"
ds = ds.cache()  # Cache data for faster access (helps in improving performance by avoiding redundant computations)
ds = ds.shuffle(60000)  # Shuffle the dataset with a buffer size of 60000 to ensure the data is randomly mixed.
ds = ds.batch(128)  # Group dataset into batches of 128 samples (useful for efficient training and processing)
ds = ds.prefetch(64)  # Prefetch next 64 batches while curr. batch is being processed (improve data pipeline efficiency)

# print(ds.as_numpy_iterator().next().shape)  # batch shape

# --- GENERATOR ---------------------------------
generator = Generator()
generator.summary()

# generate some images
img = generator.predict(np.random.randn(4, 128))  # 4 images
fig, axs = plt.subplots(1, len(img), figsize=(7, 3))
for i in range(len(img)):
    axs[i].imshow(img[i])
    axs[i].set_xticks([])
    axs[i].set_yticks([])
plt.tight_layout()
save_fig(["gan_samples_pre_training"])
# plt.show()

# --- DISCRIMINATOR -----------------------------
discriminator = Discriminator()
discriminator.summary()
predictions = discriminator.predict(img)
print(predictions)

# --- TRAINING ----------------------------------

