"""
what are filters in CNN?

Filters (or kernels) are small matrices of weights that slide over the input data to perform a convolution operation, extracting features like edges, textures, or patterns.

import torch.nn as nn

# out_channels defines the number of filters, and kernel_size defines their dimensions.
conv_layer = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)

# The filters are stored as learnable weights in the layer.
filters = conv_layer.weight

Why Pooling is used? Whats MaxPooling?
ans ->
Pooling is used to reduce the size. 

Pooling is used to reduce the spatial dimensions of the feature maps, which helps in reducing the number of parameters and computational cost. MaxPooling is a type of pooling where the maximum value in each pooling region is selected.

what us stride?
ans -> Stride is the number of pixels that the filter moves at each step. A higher stride value will result in a smaller output size, while a lower stride value will result in a larger output size.

what is padding?
ans -> Padding is the number of pixels added to the input data to make it a multiple of the kernel size. Padding is used to preserve the spatial dimensions of the feature maps.  

MaxPooling + Convolution helps in position invariant feature detection.

AveragePooling is used to reduce the size of the feature maps, which helps in reducing the number of parameters and computational cost.


BENIFITS OF POOLING
1. Reduces the number of parameters and computational cost.
2. Makes the network invariant to small translations in the input.
3. Reduces overfitting by providing a form of regularization.


Why to use data augmentation?
- CNN by itself doesn't take care of rotation and scale.
- Data augmentation helps in making the model invariant to rotation and scale.

"""