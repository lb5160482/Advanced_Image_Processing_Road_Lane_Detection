import cv2
import numpy as np


def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh_min=0, thresh_max=255):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if (orient == 'x'):
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel_absolute = np.absolute(sobel)
    scaled = np.uint8(255 * sobel_absolute / np.max(sobel_absolute))
    binary = np.zeros_like(scaled)
    binary[(scaled >= thresh_min) & (scaled <= thresh_max)] = 255

    return binary


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel_mag = np.sqrt(np.multiply(sobel_x, sobel_x) + np.multiply(sobel_y, sobel_y))
    scaled_mag = 255 * sobel_mag / np.max(sobel_mag)
    binary_output = np.zeros_like(scaled_mag)
    binary_output[(scaled_mag > thresh[0]) & (scaled_mag < thresh[1])] = 255

    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    direction = np.arctan2(sobel_y, sobel_x)
    threshed_direction = np.zeros_like(direction)
    threshed_direction[(direction > thresh[0]) & (direction < thresh[1])] = 255

    return threshed_direction


def hls_s_threshold(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    binary = np.zeros_like(s_channel)
    binary[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 255

    return binary


def hsv_s_threshold(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hls[:, :, 1]
    binary = np.zeros_like(s_channel)
    binary[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 255

    return binary


def hsv_v_threshold(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hls[:, :, 2]
    binary = np.zeros_like(s_channel)
    binary[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 255

    return binary


def perspective_transfrom(img, src_points, dst_points):
    img_size = img.shape[1::-1]
    transformation = cv2.getPerspectiveTransform(src_points, dst_points)
    perspective_transformed = cv2.warpPerspective(img, transformation, img_size)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    # aa = cv2.warpPerspective(perspective_transformed, Minv, img_size)
    # cv2.imshow('aa', aa)
    # cv2.waitKey(0)
    return perspective_transformed, Minv


def get_warpback_overlay_img(raw_img, fit_points, inverse_perspevtive_trans):
    if fit_points[0] is None or fit_points[1] is None:
        return raw_img

    left_fit_x = fit_points[0]
    right_fit_x = fit_points[1]
    ploty = fit_points[2]

    # Create an image to draw the lines on
    color_warp = np.zeros_like(raw_img).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fit_x, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, inverse_perspevtive_trans, (color_warp.shape[1], color_warp.shape[0]))
    result = cv2.addWeighted(raw_img, 1, newwarp, 0.3, 0)

    return result


def add_img_info(img, curvature, vehicle_position):
    left_lane_str = 'Left lane curvature: ' + str(float("{0:.2f}".format(curvature[0]))) + ' m'
    left_lane_str_position = int(0.05*img.shape[1]), int(0.1*img.shape[0])
    right_lane_str = 'Right lane curvature: ' + str(float("{0:.2f}".format(curvature[1]))) + ' m'
    right_lane_str_position = int(0.05 * img.shape[1]), int(0.15 * img.shape[0])
    if vehicle_position < 0 :
        car_pos_str = 'Vehicle is ' + str(
            float("{0:.2f}".format(-vehicle_position))) + ' meter left of lane center'
    else:
        car_pos_str = 'Vehicle is ' + str(
            float("{0:.2f}".format(vehicle_position))) + ' meter right of lane center'
    car_pos_str_position = int(0.05 * img.shape[1]), int(0.2 * img.shape[0])

    cv2.putText(img, left_lane_str, left_lane_str_position, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2,
                color=(255, 255, 255))
    cv2.putText(img, right_lane_str, right_lane_str_position, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2,
                color=(255, 255, 255))
    cv2.putText(img, car_pos_str, car_pos_str_position, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                thickness=2,
                color=(255, 255, 255))

    return img