"""
TENSORFLOW HUB - GET ALL PRE-TRAINED MODELS

For transfer learning, we use pre-trained models.

for image classification, we use models like VGG, ResNet, Inception, Xception, etc.

for NLP, we use models like BERT, GPT, etc.

Google's MobileNet V2 Model is a pre-trained model that is used for image classification. 1.4 million images, 1000 classes.

import tensorflow as tf

# Load the pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')

# Freeze the layers of the base model to prevent them from being updated during training
base_model.trainable = False

# Add a global average pooling layer
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Add a dense layer
x = tf.keras.layers.Dense(1024, activation='relu')(x)

# Add the final output layer
predictions = tf.keras.layers.Dense(10, activation='softmax')(x)

# Create the final model
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



"""