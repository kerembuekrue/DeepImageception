import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
from src.utils.utils import save_fig
from src.model.convolutional_neural_network.cnn import CNN
from src.data_preprocessing.data_loader import load_data_tf


# LOAD DATA --------------
data_set = "breast_cancer"
batch_size = 32
img_height = 50
img_width = 50
train_ds, val_ds = load_data_tf(data_set=data_set,
                                img_shape=(img_height, img_width),
                                batch_size=batch_size)

# VISUALIZE DATA --------------

# Names of the two classes (0 and 1)
class_names = train_ds.class_names

# Plot the first six images from the training dataset
plt.figure(figsize=(8, 5))
for images, labels in train_ds.take(1):
    for i in range(6):
        ax = plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
        plt.tight_layout()
save_fig(["training_samples"])

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
epochs = 100
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

# Paths
path = "../data/" + data_set
data_dir = pathlib.Path(path)
# example: breast cancer = False
no = list(data_dir.glob('0/*'))
# example: breast cancer = True
yes = list(data_dir.glob('1/*'))

# Select a few test images from the 'no' and 'yes' categories
test_images_paths = list(no[:30]) + list(yes[:30])  # taking 3 images from each class for demonstration

# Plot the selected test images and their predicted labels
plt.figure(figsize=(8, 5))
for i, img_path in enumerate(test_images_paths):
    # load image
    img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)  # predict label
    label = "1" if "/data/breast_cancer/1" in str(img_path) else "0"  # determine label
    plt.subplot(6, 10, i + 1)
    plt.imshow(img)
    title_color = 'red' if label != str(int(round(predictions[0][0]))) else 'black'
    plt.title(f"{label}{int(round(predictions[0][0]))}", color=title_color)
    plt.axis("off")
plt.tight_layout()
save_fig(["test_predictions"])
