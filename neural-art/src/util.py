import numpy as np
from keras import backend as K
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array

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
# TODO: img_size is kinda redundant, it must the same as preprocessed image size
def deprocess_image(x, img_size):
    img_nrows, img_ncols = img_size
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


