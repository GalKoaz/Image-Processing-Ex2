import math
import numpy as np
import cv2


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    in_signal_new = np.append(np.zeros(k_size.size - 1), np.append(in_signal, np.zeros(k_size.size - 1)))
    new_vector = np.zeros(in_signal.size + k_size.size - 1)  # make new vector with this new size
    for i in range(new_vector.size):
        new_vector[i] = np.dot(in_signal_new[i: i + k_size.size], k_size[::-1])
    return new_vector


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    new_image = np.pad(in_image, (np.flip(kernel).shape[0] // 2, np.flip(kernel).shape[1] // 2), 'edge')
    new_vector = np.zeros(in_image.shape[0], in_image.shape[1])
    for x in range(in_image.shape[0]):
        for y in range(in_image.shape[1]):
            new_vector[x, y] = np.dot((new_image[x:x + np.flip(kernel).shape[0], y:y + np.flip(kernel).shape[1]]),
                                      np.flip(kernel)).sum()
    return new_vector


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    kernel = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    x = conv2D(in_image, kernel)
    y = conv2D(in_image, kernel.transpose())
    return np.arctan(y, x), np.sqrt(np.square(x) + np.square(y))


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    kernel = np.zeros((k_size // 2, k_size // 2))
    for i in range(k_size // 2):
        for j in range(k_size // 2):
            kernel[i, j] = np.exp(-((i - k_size // 2) ** 2 + (j - k_size // 2) ** 2) / (2 * sigma ** 2)) / (
                    2 * np.pi * sigma ** 2)
    return conv2D(in_image, kernel)


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    kernel = cv2.getGaussianKernel(k_size, int(0.3 * ((k_size - 1) * 0.5 - 1) + 0.8))
    return cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: opencv solution, my implementation
    """
    return


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges usint "ZeroCrossingLOG" method
    :param img: Input image
    :return: opencv solution, my implementation
    """
    smooth = cv2.GaussianBlur(img, (5, 5), 1)
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    img_laplacian = cv2.filter2D(smooth, -1, laplacian, borderType=cv2.BORDER_REPLICATE)
    img_c = zero_cross_detection(img_laplacian)
    return img_c


def zero_cross_detection(in_image: np.ndarray) -> np.ndarray:
    """
    perform a 8 neighbour zero crossing, which implies that we shall mark the current pixel as 1,
    if the sign of intensity value corresponding to any of the 8 neighbors of current pixel is
    opposite of the sign of current pixel.
    We shall iterate over each pixel in the image and mark all such zero crossings.
    :param in_image:
    :return: zero cross - binary image
    """
    img_c = np.zeros(in_image.shape)
    for i in range(0, in_image.shape[0] - 1):
        for j in range(0, in_image.shape[1] - 1):
            if in_image[i][j] > 0:
                if in_image[i + 1][j] < 0 or in_image[i + 1][j + 1] < 0 or in_image[i][j + 1] < 0:
                    img_c[i, j] = 1
            elif in_image[i][j] < 0:
                if in_image[i + 1][j] > 0 or in_image[i + 1][j + 1] > 0 or in_image[i][j + 1] > 0:
                    img_c[i, j] = 1
    return img_c


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    return


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """

    return
