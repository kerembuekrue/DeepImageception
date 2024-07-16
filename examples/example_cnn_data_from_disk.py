import PIL
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.utils.utils import save_fig
from src.model.cnn.cnn_tensorflow import CNN

# EXPLORE DATA --------------

# Paths
path = '../data/8863'
data_dir = pathlib.Path(path)

# number of images
image_count = len(list(data_dir.glob('*/*.png')))
print("total number of images:", image_count)

# example: breast cancer = False
no = list(data_dir.glob('0/*'))
im_no = PIL.Image.open(str(no[0]))
# im_no.show()

# example: breast cancer = True
yes = list(data_dir.glob('1/*'))
im_yes = PIL.Image.open(str(no[0]))
# im_yes.show()


# LOAD DATA --------------

batch_size = 32
img_height = 50
img_width = 50

# Training data-set
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Testing data-set
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Names of the two classes (0 and 1)
class_names = train_ds.class_names
print(len(class_names), class_names)

# VISUALIZE DATA --------------

# Plot the first nine images from the training dataset
plt.figure(figsize=(8, 5))
for images, labels in train_ds.take(1):
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
        plt.tight_layout()
save_fig(["training_samples"])

# print the dimensions of the first batch of the training dataset (number of images, pixels in x, pixels in y, RGB)
for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break


# CONFIGURE DATA --------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# MODEL --------------

# Instantiate the model
model = CNN()

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# view all layers of the network
model.summary()

# Train the model
epochs = 20
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds
)

# VISUALIZE TRAINING RESULTS --------------

# accuracy
plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), history.history['accuracy'], label='Training Accuracy')
plt.plot(range(epochs), history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
# loss
plt.subplot(1, 2, 2)
plt.plot(range(epochs), history.history['loss'], label='Training Loss')
plt.plot(range(epochs), history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
# save as png
save_fig(["accuracy_loss"])

# PREDICT LABELS ON NEW DATA --------------

# Select a few test images from the 'no' and 'yes' categories
test_images_paths = list(no[:3]) + list(yes[:3])  # taking 3 images from each class for demonstration

# Plot the selected test images and their predicted labels
plt.figure(figsize=(8, 5))
for i, img_path in enumerate(test_images_paths):

    # load image
    img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # predict label
    predictions = model.predict(img_array)

    # Determine the actual label from the file path
    actual_label = "1" if "/data/8863/1" in str(img_path) else "0"

    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(f"Label: {actual_label}, Prediction: {int(round(predictions[0][0]))}")
    plt.axis("off")

plt.tight_layout()
save_fig(["test_predictions"])
