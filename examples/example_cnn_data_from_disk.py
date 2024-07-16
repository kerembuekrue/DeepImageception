import matplotlib.pyplot as plt
from src.model.cnn.cnn_tensorflow import CNN
import pathlib
import PIL
import tensorflow as tf

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
# print(len(class_names), class_names)

# VISUALIZE DATA --------------

# Plot the first nine images from the training dataset
plt.figure(figsize=(7, 7))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
        plt.tight_layout()

# print the dimensions of the first batch of the training dataset (number of images, pixels in x, pixels in y, RGB)
# for image_batch, labels_batch in train_ds:
#     print(image_batch.shape)
#     print(labels_batch.shape)
#     break


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

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')


# Evaluate the model
loss, accuracy = model.evaluate(val_ds)
print(f'Validation Accuracy: {accuracy:.4f}')

plt.show()

#
# # Visualize some predictions
# def visualize_predictions(model, generator, num_images=10):
#     x_batch, y_batch = next(generator)
#     y_pred = model.predict(x_batch)
#     y_pred_class = (y_pred > 0.5).astype("int32")
#
#     plt.figure(figsize=(15, 7))
#     for i in range(num_images):
#         plt.subplot(2, num_images // 2, i + 1)
#         plt.imshow(x_batch[i])
#         plt.title(f"True: {int(y_batch[i])}   Pred: {int(y_pred_class[i][0])}")
#         plt.tight_layout()
#         plt.axis('off')
#     plt.show()
#
#
# # Visualize predictions on a batch from the validation set
# visualize_predictions(model, val_ds)
