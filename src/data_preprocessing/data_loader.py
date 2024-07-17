import tensorflow as tf
import pathlib


def load_data_tf(data_set: str, img_shape=(256, 256), batch_size: int = 32):

    data_path = "../data/" + data_set

    data_path = pathlib.Path(data_path)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_shape,
        batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.3,
        subset="validation",
        seed=123,
        image_size=img_shape,
        batch_size=batch_size)

    return train_ds, val_ds
