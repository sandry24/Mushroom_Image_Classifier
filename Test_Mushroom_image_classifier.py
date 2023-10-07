import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from PIL import Image


model = tf.keras.models.load_model("Model(ResNet50).h5")

folder_path = 'test_images_mushrooms'
image_path = 'test_images_mushrooms/test.png'
label_file = 'mushrooms/mushrooms.txt'
target_size = (224, 224)
accs = []

with open(label_file, 'r') as file:
    labels = file.read().splitlines()

testing_directory = pathlib.Path("test_images_mushrooms")

testing_files = list(testing_directory.glob('*'))

for file in testing_files:
    image = tf.keras.preprocessing.image.load_img(file, target_size=(224, 224))
    image_array = tf.keras.utils.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)

    # image = tf.io.read_file(str(file))
    # image = tf.image.decode_image(image, channels=3)  # Assuming RGB image
    #
    # # Resize and pad the image
    # image = tf.image.resize_with_pad(image, target_size[0], target_size[1])
    # image_array = tf.keras.utils.img_to_array(image)
    # image_array = np.uint8(image_array)  # Convert to uint8
    #
    # # Display the image
    # plt.imshow(image_array)
    # plt.axis('off')
    # plt.show()
    # image_array = tf.expand_dims(image_array, 0)
    predictions = model.predict(image_array)
    score = tf.nn.softmax(predictions[0])
    print(
        "This {} most likely belongs to {} with a {:.2f} percent confidence."
        .format(file, labels[np.argmax(score)], 100 * np.max(score))
    )
    accs.append(100 * np.max(score))

print(accs)
