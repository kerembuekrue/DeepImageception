import tensorflow as tf


def load_data_tf(data_path: str, img_shape=(256, 256), batch_size: int = 8):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="training",
        label_mode='categorical',
        seed=123,
        image_size=img_shape,
        batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.3,
        subset="validation",
        label_mode='categorical',
        seed=123,
        image_size=img_shape,
        batch_size=batch_size)
    return train_ds, val_ds


if __name__ == "__main__":
    train_dataset, test_val_ds = load_data_tf('data_folder', img_shape=(256, 256), batch_size=8)
    test_dataset = test_val_ds.take(686)
    val_dataset = test_val_ds.skip(686)
