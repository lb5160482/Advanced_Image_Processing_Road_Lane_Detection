import cv2
import glob
from camera_calibration import CameraCalibration
import image_processing as imgproc
import numpy as np
import platform
from line import Line

image_size = (1280, 720)
bin_img_dict = {}

# Part 1 -- Camera calibration
calibration = CameraCalibration('camera_cal/')
calibration.calibrate()
intrinsic_mat = calibration.get_intrinsic()
dist_paras = calibration.get_distortion_paras()
print('Camera intrinsix matrix is :')
print(intrinsic_mat)
print('Camera distortion parameters are :')
print(dist_paras)
print()

# Part 2 -- Apply a distortion correction to raw images.
calibration.undistort_images()
print()

# Part 3 -- Use color transforms, gradients, etc., to create a thresholded binary image
test_image_paths = glob.glob('./test_images/*.jpg')
print('Start generating binary images from sample images...')
for test_img_path in test_image_paths:
    test_img = cv2.imread(test_img_path)
    # distortion correction
    test_img = calibration.distort_correction(test_img)
    # test_img = cv2.GaussianBlur(test_img, (5, 5), 0)

    # binary images from different processing methods
    # sobel_y_bin = imgproc.abs_sobel_thresh(test_img, orient='y', thresh_min=20, thresh_max=100)
    # mag_bin = imgproc.mag_thresh(test_img, sobel_kernel=5, thresh=(30, 100))
    # dir_bin = imgproc.dir_threshold(test_img, sobel_kernel=5, thresh=(0.7, 1.3))
    sobel_x_bin = imgproc.abs_sobel_thresh(test_img, sobel_kernel=5, orient='x', thresh_min=20, thresh_max=100)
    s_channel_bin = imgproc.hsv_s_threshold(test_img, thresh=(80, 255))
    v_channel_bin = imgproc.hsv_v_threshold(test_img, thresh=(80, 255))

    # combination of binary images
    combine = np.zeros_like(test_img[:, :, 0])
    combine[(v_channel_bin == 255) & ((sobel_x_bin == 255) | (s_channel_bin == 255))] = 255

    if platform.system() == 'Windows':
        window_name = test_img_path[test_img_path.rfind('\\') + 1:]
    else:
        window_name = test_img_path[test_img_path.rfind('/') + 1:]
    # cv2.imshow(window_name, combine)
    # cv2.waitKey(0)
    # cv2.destroyWindow(window_name)

    # file_name = './output_images/bin_' + window_name;
    # cv2.imwrite(file_name, combine)

    bin_img_dict[window_name] = combine
print('Finish generating binary images from sample images!')
print()

# Part 4 -- Apply a perspective transform to rectify binary image ("birds-eye view")
print('Start perspective transformation...')
offset = 200
perspective_src_points = np.float32(
    [[233, 694], [595, 450], [686, 450], [1073, 694]])  # These points are manually selected
perspective_dst_points = np.float32([[offset, image_size[1]], [offset, 0],
                                     [image_size[0] - offset, 0], [image_size[0] - offset, image_size[1]]])
for test_img_path in test_image_paths:
    if platform.system() == 'Windows':
        img_name = test_img_path[test_img_path.rfind('\\') + 1:]
    else:
        img_name = test_img_path[test_img_path.rfind('/') + 1:]
    test_img = cv2.imread(test_img_path)
    # distortion correction
    test_img = calibration.distort_correction(test_img)

    # file_name = './output_images/undist_' + img_name;
    # cv2.imwrite(file_name, test_img)

    # cv2.line(test_img, (233, 694), (595, 450), color=(0, 0, 255))
    # cv2.line(test_img, (686, 450), (1073, 694), color=(0, 0, 255))
    # file_name = './output_images/perepective_region.jpg'
    # cv2.imwrite(file_name, test_img)
    bird_view_img_rgb, Minv = imgproc.perspective_transfrom(test_img, perspective_src_points, perspective_dst_points)
    bird_view_img_binary, Minv = imgproc.perspective_transfrom(bin_img_dict[img_name], perspective_src_points,
                                                         perspective_dst_points)
    # cv2.imshow(img_name, bird_view_img_binary)
    # cv2.waitKey(0)
    # cv2.destroyWindow(window_name)

    cv2.imwrite('./output_images/bird_view_rgb_' + img_name, bird_view_img_rgb)
    cv2.imwrite('./output_images/bird_view_binary_' + img_name, bird_view_img_binary)

print('Perpective transformation finished!')
