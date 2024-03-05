import cv2
import numpy as np

one_byte_scale = 1.0 / 255.0


def img_crop(img_arr, top, bottom):
    if bottom is 0:
        end = img_arr.shape[0]
    else:
        end = -bottom
    return img_arr[top:end, ...]


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def applyAndPack(images, action):
    """
    Images is a colletion of pairs (`title`, image). This function applies `action` to the image part of `images`
    and pack the pair again to form (`title`, `action`(image)).
    """
    images2 = list(map(lambda img: (img[0], action(img[1])), images))
    return images2


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def normalize_and_crop(img_arr, crop_top=0, crop_bottom=0):
    img_arr = img_arr.astype(np.float32) * one_byte_scale
    if crop_top or crop_bottom:
        img_arr = img_crop(img_arr, crop_top, crop_bottom)
        if len(img_arr.shape) == 2:
            img_arr_h = img_arr.shape[0]
            img_arr_w = img_arr.shape[1]
            img_arr = img_arr.reshape(img_arr_h, img_arr_w, 1)
    return img_arr


def augment_image(img_arr, dimensions=(240, 320)):
    low_threshold = 66
    high_threshold = 233

    # definition for mask
    left = 0 * dimensions[1]
    left_bottom = right_bottom = dimensions[0] * 0.47
    right = 1 * dimensions[1]
    top = 0.54 * dimensions[0]
    top_width = 0.05 * dimensions[1]

    top_left = (left + right - top_width) / 2
    top_right = (left + right + top_width) / 2
    vertices = np.array(
        [[(left, left_bottom), (top_left, top), (top_right, top),
          (right, right_bottom), (right, dimensions[0]), (left, dimensions[0])]],
        dtype=np.int32)

    # parameters for filtering lines
    left_min = left
    left_max = 0.02 * dimensions[1]
    right_min = 0.86 * dimensions[1]
    right_max = right

#    cv2.imshow("img", img_arr)

    img = grayscale(img_arr)

    img = gaussian_blur(img, 5)
#    cv2.imshow("blur", img)

    img = canny((img * 255).astype(np.uint8), low_threshold, high_threshold)
    img = region_of_interest(img, vertices)
#    cv2.imshow("img", img)
#    cv2.waitKey(1)
    return img
