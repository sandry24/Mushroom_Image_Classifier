import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os, warnings
import logging
from tensorflow import keras
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore")
tf.get_logger().setLevel(logging.WARNING)


data_dir = 'mushrooms/data/data'
label_file = 'mushrooms/mushrooms.txt'
image_size = (512, 512)
target_image_size = (224, 224)
batch_size = 16

train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    validation_split=0.2,
    subset='both',
    seed=42,
    shuffle='True',
    image_size=image_size,
    batch_size=batch_size,
)


def resize_image(image, label):
    resized_image = tf.image.resize(image, (224, 224))
    return resized_image, label


class_names = train_ds.class_names
num_classes = len(class_names)
print(class_names)

train_ds = train_ds.map(resize_image)
val_ds = val_ds.map(resize_image)

# View the data
# plt.figure(figsize=(20, 20))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
#     plt.show()

# Verify shape
# for image_batch, label_batch in train_ds:
#     print(image_batch.shape)
#     print(label_batch.shape)
#     break

# Performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# Normalization
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# Verify normalization
# image_batch, label_batch = next(iter(normalized_ds))
# first_image = image_batch[0]
# print(np.min(first_image), np.max(first_image))

# Augment the Data
data_augmentation = keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical", input_shape=(target_image_size[0], target_image_size[1], 3)),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# View Data augmentation
# plt.figure(figsize=(20, 20))
# for images, _ in train_ds.take(1):
#     for i in range(9):
#         augmented_images = data_augmentation(images)
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(augmented_images[0].numpy().astype("uint8"))
#         plt.axis("off")
#     plt.show()

resnet_model = tf.keras.applications.resnet50.ResNet50(weights="imagenet")
last_layer = resnet_model.get_layer("avg_pool")
print(resnet_model.inputs)
resnet_layers = tf.keras.Model(inputs=resnet_model.inputs, outputs=last_layer.output)

model = tf.keras.Sequential([
    data_augmentation,
    resnet_layers,
    tf.keras.layers.Dense(num_classes, activation='softmax'),
])

model.layers[0].trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

model.summary()

# Define the Callback
checkpoint = ModelCheckpoint("Model(ResNet50).h5", monitor='val_accuracy', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto', save_freq="epoch")
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1,
                              mode='max', cooldown=2, patience=2, min_lr=0)

# Train the model
epochs = 50
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint, early, reduce_lr]
)

# Plot the results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')
plt.title('Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.title('Validation Loss')
plt.show()

# Save the model
model.save('saved_models/conv_mushrooms')
