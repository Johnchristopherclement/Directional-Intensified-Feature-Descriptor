import cv2
import numpy as np
import matplotlib.pyplot as plt
import core_routine as cr
import calculate as cl

def main():
    image = cv2.imread('D:\Augmented\Lenna.png')
    if len(image.shape) == 3 and image.shape[2] == 3:
        print("Image is in BGR format")
    else:
        print("Image is not in BGR format")

    angles = np.arange(0, 100, 10)
    repeatability_values = []

# Iterate over the angles and compute repeatability for each angle
    for angle in angles:
        ellipse_center = (250, 250)
        ellipse_axes = (150, 80)
        ellipse_angle = angle

# Compute the rotation matrix for rotating the image and ellipse
        M = cv2.getRotationMatrix2D(ellipse_center, ellipse_angle, 1.0)

# Rotate the image and ellipse
        rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        rotated_ellipse_center = cv2.transform(np.array([[ellipse_center]]), M)[0][0]

# Draw the ellipse on the original and rotated images
        cv2.ellipse(image, ellipse_center, ellipse_axes, ellipse_angle, 0, 360, (0, 255, 0), 2)
        cv2.ellipse(rotated_image, tuple(rotated_ellipse_center), ellipse_axes, ellipse_angle, 0, 360, (0, 255, 0), 2)

# Collect features from the original image
        original_features = []
        original_roi = image[ellipse_center[1] - ellipse_axes[1]:ellipse_center[1] + ellipse_axes[1],
                    ellipse_center[0] - ellipse_axes[0]:ellipse_center[0] + ellipse_axes[0]]
        original_roi_gray = cv2.cvtColor(original_roi, cv2.COLOR_BGR2GRAY)

# Apply Prewitt filter to the original ROI
        gradient_magnitude, gradient_direction = cr.get_grad(original_roi_gray)

# Compute HOG feature descriptor for the original ROI
        hog_features = cr.compute_hog(original_roi_gray,gradient_magnitude, gradient_direction,9,(8,8), (3,3),'l1',False, False,True,True)

# Append the gradient magnitude, gradient direction, and HOG features to original_features list
  #original_features.append(gradient_magnitude)
  #original_features.append(gradient_direction)
        original_features.append(hog_features)

# Collect features from the rotated image
        rotated_features = []
        rotated_roi = rotated_image[int(rotated_ellipse_center[1]) - ellipse_axes[1]:int(rotated_ellipse_center[1]) + ellipse_axes[1],
                            int(rotated_ellipse_center[0]) - ellipse_axes[0]:int(rotated_ellipse_center[0]) + ellipse_axes[0]]
        rotated_roi_gray = cv2.cvtColor(rotated_roi, cv2.COLOR_BGR2GRAY)

        rotated_gradient_magnitude, rotated_gradient_direction = cr.get_grad(rotated_roi_gray)

        rotated_hog_features = cr.compute_hog(original_roi_gray,rotated_gradient_magnitude, rotated_gradient_direction,9,(8,8), (3,3),'l1',False, False,True,True)

        rotated_features.append(rotated_hog_features)

        repeatability = cl.calculate_repeatability(original_features[0], rotated_features[0])
        repeatability_values.append(repeatability)


    plt.plot(angles, repeatability_values)
    plt.xlabel("Rotation Angle (degrees)")
    plt.ylabel("Repeatability")
    plt.title("Repeatability vs. Rotation Angle")
    plt.show()

# Load the image
    image = cv2.imread('D:\Augmented\Lenna.png', cv2.IMREAD_GRAYSCALE)
    gradient_magnitude, gradient_direction = cr.get_grad(image)

# Compute HOG features for the entire image with horizontal and vertical details
    hog_features_horizontal_vertical, hog_image_horizontal_vertical = cr.compute_hog(image,gradient_magnitude, gradient_direction,9,(8,8), (3,3),'l1',True, False,True,True)

# Draw the entire image with the features overlapped (horizontal, vertical, and diagonal details)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap='gray')
    ax.imshow(hog_image_horizontal_vertical, cmap='gray', alpha=0.8)
#draw_hog_features(hog_features_all_directions, image.shape[1], image.shape[0], ax)
    ax.set_title('')
    ax.axis('off')
    plt.show()

main()