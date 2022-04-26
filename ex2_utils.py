import math
import numpy as np
import cv2


# ----------------------------------------------------------------------------
# Created By : Gal Koaz
# Created Date : 24-04-2022
# Python version : '3.8'
# ---------------------------------------------------------------------------

def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 206260168


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    """
    We Created a new signal with the zero array of k_size - 1 and signal , zero array with k_size - 1
    In addition we make a new vector with zero array of that size of both signal and k_size
    for each i in the vector size we add to the place i the product of the new signal of i + size 
    and the last element k_size of the array,
    :return: new_vector for the conv1D function.
    """
    in_signal_new = np.append(np.zeros(k_size.size - 1), np.append(in_signal, np.zeros(k_size.size - 1)))
    new_vector = np.zeros(in_signal.size + k_size.size - 1)
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
    """
    First we created a pad: numpy.pad() function is used to pad the Numpy arrays. Sometimes there is a need to
    perform padding in Numpy arrays, then numPy.pad() function is used. The function returns the padded array of rank
    equal to the given array and the shape will increase according to pad_width. and created a new array of zeros
    with the size of the in_image dimensions, for y and x in the in_image we insert to the output array the x of the
    flip kernel + y and the x of the flip kernel multiply the flip of the kernel all together with the sum.
    """
    image_padded = np.pad(in_image, ((np.flip(kernel)).shape[0] // 2, (np.flip(kernel)).shape[1] // 2), 'edge')
    output = np.zeros((in_image.shape[0], in_image.shape[1]))
    for y in range(in_image.shape[0]):
        for x in range(in_image.shape[1]):
            output[y, x] = (image_padded[y:y + (np.flip(kernel)).shape[0], x:x + (np.flip(kernel)).shape[1]] * np.flip(
                kernel)).sum()
    return output


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    """
    We used the kernel [[0, 0, 0], [-1, 0, 1], [0, 0, 0]], with the x of this kernel and the transpose of y,
    we return the artan of y,x and the sqrt of squares x + squares y
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
    """First we use the sigma calculation of 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8 we made a new kernel of array zeros 
    with the size x,y -> k_size // 2 we loop over the x,y in the range of k_size // 2 both, and we update in every 
    kernel x,y the exp of -((x - k_size // 2) ** 2 + (y - k_size // 2) ** 2) / (2 * sigma ** 2)) / ( 2 * np.pi * 
    sigma ** 2), and return the conv2D with the in_image and the new kernel we made.
    """
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    kernel = np.zeros((k_size // 2, k_size // 2))
    for x in range(k_size // 2):
        for y in range(k_size // 2):
            kernel[x, y] = np.exp(-((x - k_size // 2) ** 2 + (y - k_size // 2) ** 2) / (2 * sigma ** 2)) / (
                    2 * np.pi * sigma ** 2)
    return conv2D(in_image, kernel)


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    """
    we made a new kernel with of the cv2 GaussianKernel of the k_size and the sigma formula,
    then we return the cv2 filter2D with the in_image and the kernel we made.
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
    """we use cv2 GaussianBlur to make the image smooth, then we used the laplacian kernel, we apply the cv2 filter2D 
    with the smooth and the laplacian kernel then we used the helper function of zero_cross_detection
    then we return the img_c with edgeDetectionZeroCrossingLOG as we required.
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
    """
    First we multiply the image with 255 to get the original image, after we use the cv2 canny, we crated a new 
    array with zeros of the 3d dimensions of imgCanny x, imgCanny y, and the abs value of the max radius - min radius 
    we created a direction of the arctan with angel of sobel of -1,0,1 and the -1,1,0 of the img with the y/x, 
    we over each pixel in the image to detected a potential circles and every pixel x,y we over of the range of the 
    radius, to make sure we dont miss any circle in the image, after that if we detected a good circle we add him to 
    the array we have created, and in the last we create a new lst we need to return, when we over the circle list 
    and choose the circles we threshold of 23 means that we do not need multiply circles and we return the list of
    x,y and radius of the circle.
    """
    img *= 255
    imgCanny = cv2.Canny(img.astype(np.uint8), 100, 200)
    h_c = np.zeros((imgCanny.shape[0], imgCanny.shape[1], max_radius - min_radius))
    d = np.arctan2(cv2.Sobel(img, -1, 0, 1), cv2.Sobel(img, -1, 1, 0))
    for j in range(0, imgCanny.shape[1]):
        for i in range(0, imgCanny.shape[0]):
            for r in range(h_c.shape[2], h_c.shape[2] + min_radius):
                if imgCanny[i, j] != 0:
                    try:
                        h_c[int(j + r * np.cos(d[i, j])), int(i + r * np.sin(d[i, j])), r - min_radius] += 1
                        h_c[int(j - r * np.cos(d[i, j])), int(i - r * np.sin(d[i, j])), r - min_radius] += 1
                    except IndexError as e:
                        pass
    lst = []
    for r in range(h_c.shape[2]):
        for x in range(0, img.shape[0]):
            for y in range(0, img.shape[1]):
                if h_c[x, y, r] > 23:
                    lst.append((x, y, min_radius + r))
    return lst


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    """
    First,  we make the each pixel to be float32, we create new zero array of the in_image dimension , we used the 
    GaussianKernel with the k_size and the sigma space, we saved the dimensions of the in_image. we over the x and y 
    of the image when the range is k_size // 2 and dimension x - dimension y // 2 in each x and y we save the element 
    in x + k_size // 2 + 1, y + k_size // 2 + 1, then we used this array - the place of the x,y -> k_size // 2 then 
    we used the sigma calculated to each element we multiply the weights of the gauss kernel and the array we 
    calculated then we sum the values divide the weights then we multiply back with 255 then change the elements to 
    uint8 and return the image as we required.
    All implements is based on the lecture we have learn and shows in the moodle.
    """
    open_cv = cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space)
    image = np.zeros(in_image.shape)
    padded_image = cv2.copyMakeBorder(in_image, math.floor(k_size / 2), math.floor(k_size / 2), math.floor(k_size / 2),
                                      math.floor(k_size / 2), cv2.BORDER_REPLICATE, None, value=0)
    gaussKer = cv2.getGaussianKernel(k_size, sigma_space)
    gaussKer = gaussKer.dot(gaussKer.T)
    for x in np.arange(0, in_image.shape[0], 1).astype(int):
        for y in np.arange(0, in_image.shape[1], 1).astype(int):
            pivot_v = in_image[x, y]
            neighbor_hood = padded_image[x:x + k_size, y:y + k_size]
            diff = pivot_v - neighbor_hood
            diff_gau = np.exp(-np.power(diff, 2) / (2 * sigma_color))
            combo = gaussKer * diff_gau
            result = combo * neighbor_hood / combo.sum()
            ans = result.sum()
            image[x][y] = ans
    return open_cv, image
