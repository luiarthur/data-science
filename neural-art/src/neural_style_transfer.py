import argparse
### Parse arguments ############################################################
parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('content_image_path', type=str)
parser.add_argument('style_image_path', type=str)
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')
parser.add_argument('--iter', type=int, default=10, required=False,
                    help='Number of iterations to run.')
parser.add_argument('--content_weight', type=float, default=0.025, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
                    help='Total Variation weight.')
args = parser.parse_args()
### End of Parse arguments #####################################################

from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications import vgg19
from keras import backend as K

from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import time

### PREPROCESSING FUNCTIONS ####################################################
# util function to open, resize, and format pictures into appropriate tensors
def preprocess_image(image_path, target_size):
    """
    Example:
    import matplotlib.pyplot as plt
    preprocessed_img = preprocess_image(path_to_image, (400,400))
    plt.imshow(preprocessed_img[0])
    plt.show()
    """
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) # adds dimension to axis 0: (h,w,c) -> (n,h,w,c)
    img = vgg19.preprocess_input(img) # depending on mode, will rescale images
    return img


# util function to convert a tensor into a valid image	
def deprocess_image(x, image_size):
    img_nrows, img_ncols = image_size
    if K.image_dim_ordering() == 'th':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
### END OF PREPROCESSING FUNCTIONS ###########################################

### LOSS FUNCTIONS ###########################################################
def content_loss(content_image, generated_image):
    return K.sum(K.square(content_image - generated_image))

# the gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        # batch_flatten turns tensor into 2D array
        # while preserving the first dimension
        filters = K.batch_flatten(x)
    else:
        # channel is last
        filters = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(filters, K.transpose(filters))
    return gram

def style_loss(style_image, generated_image, target_size): 
    #assert K.ndims(style_image) == 3 # TODO: why???
    #assert K.ndims(generated_image) == 3 # TODO: why???
    assert K.ndims(style_image) == K.ndims(generated_image) # TODO: is what I wrote correct?
    assert K.shape(style_image) == K.shape(generated_image) # TODO: is what I wrote correct?

    gm_style = gram_matrix(style_image)
    gm_generated = gram_matrix(generated_image)

    #channels = 3 # TODO: why???
    #size = target_size[0] * target_size[1]
    #CONST = channels * size
    CONST = K.prod(K.shape(style_image))

    return K.sum(K.squre(gm_style - gm_generated)) / ((2. * CONST) ** 2)


### total variation loss encourages smoothness of generated image
### by penalizing more for images where neighbors are not as similar to itself
def total_variation_loss(x, target_size, temperature=1.25):
    assert K.ndim(x) == 4
    H, W = target_size
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, H-1, W-1] - x[:, :, 1:, W-1])
        b = K.square(x[:, :, H-1, W-1] - x[:, :, H-1, 1:])
    else:
        # channels_last
        a = K.square(x[:, H-1, W-1, :] - x[:, 1:, W-1, :])
        b = K.square(x[:, H-1, W-1, :] - x[:, H-1, 1:, :])

    return K.sum(K.pow(a + b, temperature))


### END OF LOSS FUNCTIONS ####################################################

if __name__ == "__main__":
    # TODO:
    #loss = K.variable(0.)
    #outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    #layer_features = outputs_dict['block5_conv2']
    #content_image_filters = layer_features[0, :, :, :]
    #generated_image_filters = layer_features[2, :, :, :]
    #loss += content_weight * content_loss(base_image_features,
    #                                      combination_features)

f
