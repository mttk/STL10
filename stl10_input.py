import numpy as np

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the binary file with image data
DATA_PATH = './data/train_X.bin'

def read_all_images(path_to_data):
    '''
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    '''

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow

        images = np.transpose(images, (0, 3, 2, 1))
        return images

def read_single_image(image_file):
    '''
    CAREFUL! - this method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    '''
    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    # transpose to standard format
    image = np.transpose(image, (2, 1, 0))
    return image

def plot_image(image):
    '''
    :param image: the image to be plotted in a 3-D matrix format
    :return: None
    '''
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":

    # test to check if the image is read correctly
    with open(DATA_PATH) as file:
        image = read_single_image(file)
        plot_image(image)

    # test to check if the whole dataset is read correctly
    images = read_all_images(DATA_PATH)
    print images.shape
