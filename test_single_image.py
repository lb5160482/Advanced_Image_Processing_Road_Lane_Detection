import cv2
from camera_calibration import CameraCalibration
import image_processing as imgproc
import numpy as np
from line import Line

scaled_size = 1
image_size = (int(1280 * scaled_size), int(720 * scaled_size))
offset = image_size[1] * 0.3

file_name = './output_images/undist_straight_lines1.jpg'

calibration = CameraCalibration('camera_cal/')
calibration.calibrate()
intrinsic_mat = calibration.get_intrinsic()
dist_paras = calibration.get_distortion_paras()

perspective_src_points = scaled_size * np.float32(
    [[233, 694], [595, 450], [686, 450], [1073, 694]])  # These points are manually selected
perspective_dst_points = np.float32([[offset, image_size[1]], [offset, 0],
                                     [image_size[0] - offset, 0], [image_size[0] - offset, image_size[1]]])

#######################
img = cv2.imread(file_name)
undist = calibration.distort_correction(img)
undist = cv2.resize(undist, (0, 0), fx=scaled_size, fy=scaled_size)

hls = cv2.cvtColor(undist, cv2.COLOR_BGR2HSV)
h = hls[:, :, 0]
s = hls[:, :, 1]
v = hls[:, :, 2]
s_bin = np.zeros_like(s)
s_bin[(s >= 100) & (s <= 255)] = 255
v_bin = np.zeros_like(v)
v_bin[(v >= 80) & (v <= 255)] = 255

gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
gray_bin = np.zeros_like(gray)

gray_bin[gray > 70] = 255
sobel_x_bin = imgproc.abs_sobel_thresh(undist, orient='x', sobel_kernel=5, thresh_min=20, thresh_max=100)
sobel_y_bin = imgproc.abs_sobel_thresh(undist, orient='y', thresh_min=20, thresh_max=100)
mag_bin = imgproc.mag_thresh(undist, sobel_kernel=5, thresh=(30, 100))
dir_bin = imgproc.dir_threshold(undist, sobel_kernel=5, thresh=(0.7, 1.3))
s_channel_bin = imgproc.hsv_s_threshold(undist, thresh=(80, 255))

combine = np.zeros_like(sobel_x_bin)
combine[(v_bin == 255) & ((sobel_x_bin == 255) | (s_channel_bin == 255))] = 255

bird_view_img_binary = imgproc.perspective_transfrom(combine, perspective_src_points, perspective_dst_points)

left_line = Line(image_size)
left_line.find_line(bird_view_img_binary)

# cv2.imshow('bird_view_img_binary', bird_view_img_binary)
# cv2.waitKey(0)

cv2.destroyAllWindows()