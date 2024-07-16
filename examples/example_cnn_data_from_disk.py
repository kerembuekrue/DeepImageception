from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from src.model.cnn.cnn_tensorflow import CNN

# Paths
data_dir = '../data/8863'
batch_size = 32
img_height = 50
img_width = 50


# Create ImageDataGenerator for training and validation sets
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)


# Instantiate the model
model = CNN()

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation Accuracy: {accuracy:.4f}')


# Visualize some predictions
def visualize_predictions(model, generator, num_images=10):
    x_batch, y_batch = next(generator)
    y_pred = model.predict(x_batch)
    y_pred_class = (y_pred > 0.5).astype("int32")

    plt.figure(figsize=(15, 7))
    for i in range(num_images):
        plt.subplot(2, num_images // 2, i + 1)
        plt.imshow(x_batch[i])
        plt.title(f"True: {int(y_batch[i])}   Pred: {int(y_pred_class[i][0])}")
        plt.tight_layout()
        plt.axis('off')
    plt.show()


# Visualize predictions on a batch from the validation set
visualize_predictions(model, validation_generator)
