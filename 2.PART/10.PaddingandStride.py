"""
What is Padding and Stride?

Padding -> The process of adding extra pixels (usually zeros) around the input image to control the output size and preserve edge information.

Stride -> The number of pixels the filter moves at each step.


What is Valid Convolution and Same Convolution?
Valid Convolution -> A convolution operation where no padding is applied, resulting in an output size smaller than the input (Output = Input - Filter + 1).

Same Convolution -> A convolution operation where padding is added to the input such that the output size is the same as the input size (Output = Input).

Example of some Tenforflow CNN layer with params and explanation:

import tensorflow as tf
from tensorflow.keras import layers

# Create a simple CNN model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# In the first layer, the padding is 'valid' (default) and stride is (1, 1) (default).
# The output size is calculated as (Input - Filter + 1) = (28 - 3 + 1) = 26.
# This results in an output shape of (26, 26, 32).

# In the second layer, the padding is 'valid' (default) and stride is (2, 2) (default).
# The output size is calculated as (Input - Filter + 1) = (26 - 3 + 1) = 24.
# This results in an output shape of (24, 24, 64).

# In the third layer, the padding is 'valid' (default) and stride is (2, 2) (default).
# The output size is calculated as (Input - Filter + 1) = (24 - 3 + 1) = 22.
# This results in an output shape of (22, 22, 64).

# Print the model summary
model.summary()

"""