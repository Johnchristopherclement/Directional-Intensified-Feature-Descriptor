import cv2
import numpy as np
import matplotlib.pyplot as plt
import core_routine as cr
import calculate as cl
import scipy

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(src, table)

def rotation_repeatability_score(image):
    window_size = 10

    # Generate hexagonal sliding window indices

    # Define the angles for rotation
    angles = np.arange(10, 100, 10)

    # Lists to store repeatability scores
    repeatability_hv = []
    repeatability_hvd = []

    # Compute HOG features and repeatability scores for each angle
    for angle in angles:
        # Rotate the image
        M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        # gradient_magnitude, gradient_direction = cr.cr.get_grad(image)
        weights,theta = cr.get_grad(image)

        # Compute HOG features for the original and rotated images with horizontal and vertical details
        hog_features_hv_orig = cr.compute_hog(image,weights,theta, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
                                           block_norm='l1', visualise=False, transform_sqrt=False, feature_vector=True,
                                           verbose=True)
        hog_features_hv_rot = cr.compute_hog(rotated_image,weights,theta, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
                                          block_norm='l1', visualise=False, transform_sqrt=False, feature_vector=True,
                                          verbose=True)

        # Compute repeatability score for horizontal and vertical details
        repeatability_hv.append(cl.calculate_repeatability(hog_features_hv_orig, hog_features_hv_rot))

        # Compute HOG features for the original and rotated images with horizontal, vertical, and diagonal details
        weights,theta = cr.get_grad(image)
        #gradient_magnitude, gradient_direction = cr.get_grad(image)
        hog_features_hvd_orig = cr.compute_hog(image,weights,theta, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
                                            block_norm='l1', visualise=False, transform_sqrt=False, feature_vector=True,
                                            verbose=True)
        hog_features_hvd_rot = cr.compute_hog(rotated_image,weights,theta, orientations=9, pixels_per_cell=(8, 8),
                                           cells_per_block=(3, 3), block_norm='l1', visualise=False,
                                           transform_sqrt=False, feature_vector=True, verbose=True)

        # Compute repeatability score for horizontal, vertical, and diagonal details
        repeatability_hvd.append(cl.calculate_repeatability(hog_features_hvd_orig, hog_features_hvd_rot))
    return angles, repeatability_hv, repeatability_hvd


def blur_repeatability_score(image):
    # gradient_magnitude, gradient_direction = cr.get_grad(image)

    # Generate hexagonal sliding window indices

    # hog_features_orig = hog(gradient_magnitude, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, multichannel=False)
    repeatability_compress = []
    repeatability_hvd = []
    blr = range(1, 10, 1)
    for i in range(len(blr)):
        window_size = 10
        # from skimage.transform import rescale
        blurred_image = scipy.ndimage.gaussian_filter(image, sigma=blr[i])
        # gradient_magnitude, gradient_direction = cr.get_grad(image)
        weights, theta = cr.get_grad(image)
        # Generate hexagonal sliding window indices

        hog_features_hv_orig = cr.compute_hog(image, weights, theta, orientations=2, pixels_per_cell=(8, 8),
                                           cells_per_block=(2, 2), block_norm='l1', visualise=False,
                                           transform_sqrt=False, feature_vector=True,
                                           verbose=True)
        hog_features_hv_compress = cr.compute_hog(blurred_image, weights, theta, orientations=2, pixels_per_cell=(8, 8),
                                               cells_per_block=(2, 2), block_norm='l1', visualise=False,
                                               transform_sqrt=False, feature_vector=True,
                                               verbose=True)

        repeatability_compress.append(cl.calculate_repeatability(hog_features_hv_orig, hog_features_hv_compress))
        weights, theta = cr.get_grad(image)
        hog_features_hvd_orig = cr.compute_hog(image, weights, theta, orientations=2, pixels_per_cell=(8, 8),
                                            cells_per_block=(2, 2), block_norm='l1', visualise=False,
                                            transform_sqrt=False, feature_vector=True,
                                            verbose=True)
        hog_features_hvd_compress = cr.compute_hog(blurred_image, weights, theta, orientations=2, pixels_per_cell=(8, 8),
                                                cells_per_block=(2, 2), block_norm='l1', visualise=False,
                                                transform_sqrt=False, feature_vector=True,
                                                verbose=True)
        repeatability_hvd.append(cl.calculate_repeatability(hog_features_hvd_orig, hog_features_hvd_compress))
    return blr, repeatability_compress, repeatability_hvd

def light_variation_repeatability_score(image):
    weights,theta = cr.get_grad(image)
    repeatability_compress = []
    repeatability_hvd = []
    lgt = range(1, 10, 1)
    for i in range(len(lgt)):
        window_size = 10
        image_rescaled = gammaCorrection(image, gamma=lgt[i])
        hog_features_hv_orig = cr.compute_hog(image,weights,theta, orientations=2, pixels_per_cell=(8, 8), cells_per_block=(2, 2),block_norm='l1', visualise=False, transform_sqrt=False, feature_vector=True,
                                           verbose=True)
        hog_features_hv_compress = cr.compute_hog(image_rescaled,weights,theta, orientations=2, pixels_per_cell=(8, 8),
                                               cells_per_block=(2, 2),block_norm='l1', visualise=False, transform_sqrt=False, feature_vector=True,
                                           verbose=True)
        # Compute gradient orientations using Sobel filters
        #gradient_magnitude, gradient_direction = cr.get_grad(image_rescaled)
        weights,theta = cr.get_grad(image_rescaled)

        repeatability_compress.append(cl.calculate_repeatability(hog_features_hv_orig, hog_features_hv_compress))
        hog_features_hvd_orig = cr.compute_hog(image,weights,theta, orientations=2, pixels_per_cell=(8, 8), cells_per_block=(2, 2),block_norm='l1', visualise=False, transform_sqrt=False, feature_vector=True,
                                           verbose=True)
        hog_features_hvd_compress = cr.compute_hog(image_rescaled,weights,theta, orientations=2, pixels_per_cell=(8, 8),
                                                cells_per_block=(2, 2),block_norm='l1', visualise=False, transform_sqrt=False, feature_vector=True,
                                           verbose=True)
        repeatability_hvd.append(cl.calculate_repeatability(hog_features_hvd_orig, hog_features_hvd_compress))
    return lgt, repeatability_compress, repeatability_hvd

def get_feat(image):
    hog_image_col = image
    for ij in range(0, 3):
        weights, theta = cr.get_grad(image[:, :, ij])

        # Compute HOG features for the entire image with horizontal and vertical details
        features, hog_image = cr.compute_hog(image[:, :, ij],weights,theta, orientations=9, pixels_per_cell=(1, 1),
                                      cells_per_block=(2, 2), block_norm='l1', visualise=True, transform_sqrt=False,
                                      feature_vector=True, verbose=True)
        hog_image_col[:, :, ij] = hog_image
    return hog_image_col


def main():
    image = cv2.imread('Lenna.png')
    # image = cv2.resize(image, (32,32))
    rot_hog_features_hvd_orig = get_feat(image)
    ang = 30
    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), ang, 1.0)
    rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    rot_hog_features_hvd_rot = get_feat(rotated_image)
    [angles, rot_repeatability_hv, rot_repeatability_hvd] = rotation_repeatability_score(image)

    blurr = 0.1
    blurred_image = scipy.ndimage.gaussian_filter(image, sigma=blurr)
    blur_hog_features_hvd_orig = get_feat(blurred_image)

    [blr, blur_repeatability_hv, blur_repeatability_hvd] = blur_repeatability_score(image)

    lght = 2
    image_rescaled = gammaCorrection(image, gamma=lght)
    light_hog_features_hvd_orig = get_feat(image_rescaled)
    [lgt, light_repeatability_hv, light_repeatability_hvd] = light_variation_repeatability_score(image)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    ax1.imshow(rot_hog_features_hvd_orig, cmap='gray', alpha=0.8)
    ax1.set_title('features of an Rotated Image')
    ax1.axis('off')
    plt.subplots_adjust(left=0.1)
    ax2.plot(angles, rot_repeatability_hv, marker="*", label='TB FILTER')
    ax2.plot(angles, rot_repeatability_hvd, marker="<", label='TBD FILTER')
    ax2.axis([10, 60, 0.6, 1.1])
    ax2.set_xlabel("Rotation Angle (degrees)")
    ax2.set_ylabel("Repeatability score of rotated image")
    ax2.set_title("Repeatability vs. Rotation Angle")
    ax2.legend(["TB FILTER", "TBD FILTER"])
    fig, (axs1, axs2) = plt.subplots(1, 2)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    axs1.imshow(blur_hog_features_hvd_orig, cmap='gray', alpha=0.8)
    axs1.set_title('Features of an Blur Image')
    axs1.axis('off')
    axs2.plot(blr, blur_repeatability_hv, marker="*", label='TB FILTER')
    axs2.plot(blr, blur_repeatability_hvd, marker="<", label='TBD FILTER')
    axs2.axis([1, 10, 0.6, 1.1])

    axs2.set_xlabel("Blur Variation")
    axs2.set_ylabel("Repeatability score of blur image")
    axs2.set_title("Repeatability vs. Blur Value")
    axs2.legend(["TB FILTER", "TBD FILTER"])
    fig, (axe1, axe2) = plt.subplots(1, 2)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    axe1.imshow(light_hog_features_hvd_orig, cmap='gray', alpha=0.8)
    axe1.set_title('Features of an light variation')
    axe1.axis('off')

    axe2.plot(lgt, light_repeatability_hv, marker="*", label='TB FILTER')
    axe2.plot(lgt, light_repeatability_hvd, marker="<", label='TBD FILTER')
    axe2.axis([1, 10, 0.6, 1.1])

    axe2.set_xlabel("Light Intensity Variation in Luminance")
    axe2.set_ylabel("Repeatability score of light variation")
    axe2.set_title("Repeatability vs. Light variation")
    axe2.legend(["TB FILTER", "TBD FILTER"])
    plt.show()

main()