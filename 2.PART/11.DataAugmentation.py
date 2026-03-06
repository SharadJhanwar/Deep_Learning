"""
Explain it , Need, benifits  Give code exmples( Tensorflow flower dataset)

import tensorflow as tf
from tensorflow.keras import layers
import pathlib

# 1. Download the flower dataset
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

# 2. Define Data Augmentation layers
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.1),
  layers.RandomContrast(0.1),
])

# 3. Apply Augmentation in a Model
batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

model = tf.keras.Sequential([
  # Preprocessing layers
  layers.Input(shape=(img_height, img_width, 3)),
  data_augmentation,
  layers.Rescaling(1./255),
  
  # CNN layers
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(5) # 5 classes of flowers
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds, epochs=10)

What we can do in data augmentation?
1. RandomFlip
2. RandomRotation
3. RandomZoom
4. RandomContrast
5. RandomTranslation
6. RandomShear
7. RandomBrightness
8. RandomContrast 
9. RandomCrop

"""