import matplotlib.pyplot as plt
import numpy as np


def tile_images(images: np.ndarray,  no_across: np.int = None, no_down: np.int = None,
                show: bool = False, file_name: str = None) -> np.ndarray:
    """
    Take as set of images in, and tile them.
    Input images are represented as numpy array with 3 or 4 dims:
    shape[0]: Number of images
    shape[1] + shape[2]: size of image
    shape[3]: If > 1, then this is the color channel

    Images: The np array with images
    no_across/no_down: Force layout of subfigs. If both arte none, we get a "semi-square" image
    show: do plt.show()
    filename: If not None we save to this filename. Assumes it is fully extended (including .png or whatever)
    """

    no_images = images.shape[0]

    if no_across is None and no_down is None:
        width = int(np.sqrt(no_images))
        height = int(np.ceil(float(no_images) / width))
    elif no_across is not None:
        width = no_across
        height = int(np.ceil(float(no_images) / width))
    else:
        height = no_down
        width = int(np.ceil(float(no_images) / height))

    if len(images.shape) < 4:
        images = np.expand_dims(images, axis=-1)
    color_channels = images.shape[3]

    # Rescale
    images = images - np.min(np.min(np.min(images, axis=(1, 2))))
    images = images / np.max(np.max(np.max(images, axis=(1, 2))))

    # Build up tiled representation
    image_shape = images.shape[1:3]
    tiled_image = np.zeros((height * image_shape[0], width * image_shape[1], color_channels), dtype=images.dtype)
    for index, img in enumerate(images):
        i = int(index / width)
        j = index % width
        tiled_image[i * image_shape[0]:(i + 1) * image_shape[0], j * image_shape[1]:(j + 1) * image_shape[1], :] = \
            img          # used to be img[:, :, 0]

    plt.Figure()
    if color_channels == 1:
        plt.imshow(tiled_image[:, :, 0], cmap="binary")
    else:
        plt.imshow(tiled_image.astype(np.float32))
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)

    if show is True:
        plt.show()

    # Clean up
    plt.clf()
    plt.cla()

    return tiled_image
