import numpy as np
import cv2


class Line():
    """
    img_size: width, height
    """
    def __init__(self, img_size, scale):
        self.detected = False
        self.recent_xfit = []
        self.best_x = None
        self.best_fit = None
        self.best_left_fit_x = None
        self.best_right_fit_x = None
        self.current_fit = []
        # radius of curvature in m for current x, y in m
        self.radius_of_curvature = []
        self.diffs = None
        self.img_size = img_size
        # fit line image for visualization
        self.out_img = None
        # standard y coordinates
        self.ploty = np.linspace(0, img_size[1] - 1, img_size[1], dtype=np.int32)
        # most recent valid curvatures
        self.cur_curvature = []
        # detected left&right x, y
        self.left_x = None
        self.left_y = None
        self.right_x = None
        self.right_y = None
        self.left_fit_x = None
        self.right_fit_x = None
        # scale pixel->meter
        self.ym_per_pix = 30 / 720 / scale
        self.xm_per_pix = 3.7 / 700 / scale
        # car position offset
        self.car_pos = 0

    def find_line(self, binary_warped):
        # find lanes
        if not self.detected:
            self.blind_search(binary_warped)
        else:
            self.track_lines(binary_warped)

        # sanity check
        if self.sanity_check():
            self.detected = True
            self.best_fit = self.current_fit
        else:
            self.detected = False
        # update cars position
        if self.best_fit is not None:
            self.update_car_position()
            self.best_left_fit_x = self.left_fit_x
            self.best_right_fit_x = self.right_fit_x

        return self.out_img, [self.best_left_fit_x, self.best_right_fit_x, self.ploty], self.cur_curvature, self.car_pos

    # Note: binary_warped will be only half on the image
    def blind_search(self, binary_warped):
        #### visualization ####
        self.out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        #### visualization ####

        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        midpoint = np.int(histogram.shape[0] // 2)
        left_base_x = np.argmax(histogram[:midpoint])
        right_base_x = np.argmax(histogram[midpoint:]) + midpoint

        nwindow = 9
        window_height = np.int(binary_warped.shape[0] // nwindow)

        nonezero = binary_warped.nonzero()
        nonezero_y = np.array(nonezero[0])
        nonezero_x = np.array(nonezero[1])

        left_x_current = left_base_x
        right_x_current = right_base_x

        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        left_lane_inds = []
        right_lane_inds = []

        for window_ind in range(nwindow):
            win_y_low = binary_warped.shape[0] - (window_ind + 1) * window_height
            win_y_high = binary_warped.shape[0] - window_ind * window_height
            win_x_left_low = left_x_current - margin
            win_x_left_high = left_x_current + margin
            win_x_right_low = right_x_current - margin
            win_x_right_high = right_x_current + margin

            #### visualization ####
            cv2.rectangle(self.out_img, (win_x_left_low, win_y_low), (win_x_left_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(self.out_img, (win_x_right_low, win_y_low), (win_x_right_high, win_y_high), (0, 255, 0), 2)
            #### visualization ####

            good_left_inds = ((nonezero_y >= win_y_low) & (nonezero_y <= win_y_high) &
                              (nonezero_x >= win_x_left_low) & (nonezero_x <= win_x_left_high)).nonzero()[0]
            good_right_inds = ((nonezero_y >= win_y_low) & (nonezero_y <= win_y_high) &
                               (nonezero_x >= win_x_right_low) & (nonezero_x <= win_x_right_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                left_x_current = np.int(np.mean(nonezero_x[good_left_inds]))
            if len(good_right_inds) > minpix:
                right_x_current = np.int(np.mean(nonezero_x[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        self.left_x = nonezero_x[left_lane_inds]
        self.left_y = nonezero_y[left_lane_inds]
        self.right_x = nonezero_x[right_lane_inds]
        self.right_y = nonezero_y[right_lane_inds]

        # get fitting
        left_fit = np.polyfit(self.left_y, self.left_x, 2)
        right_fit = np.polyfit(self.right_y, self.right_x, 2)
        # update fitting coefficients
        self.current_fit = [left_fit, right_fit]

        # fitting
        self.left_fit_x = (left_fit[0] * self.ploty ** 2 + left_fit[1] * self.ploty + left_fit[2]).astype(np.int32)
        left_pts = np.vstack((self.left_fit_x, self.ploty)).T.reshape(-1, 1, 2)
        self.right_fit_x = (right_fit[0] * self.ploty ** 2 + right_fit[1] * self.ploty + right_fit[2]).astype(np.int32)
        right_pts = np.vstack((self.right_fit_x, self.ploty)).T.reshape(-1, 1, 2)

        self.out_img = cv2.polylines(self.out_img, [left_pts], False, (0, 0, 255), thickness=2)
        self.out_img = cv2.polylines(self.out_img, [right_pts], False, (0, 0, 255), thickness=2)

        #### visualization ####
        # cv2.imshow('out', self.out_img)
        # cv2.waitKey(0)
        #### visualization ####

        return left_fit, right_fit

    def track_lines(self, binary_warped):
        #### visualization ####
        self.out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        #### visualization ####

        nonezero = binary_warped.nonzero()
        nonezero_y = nonezero[0]
        nonezero_x = nonezero[1]
        margin = 100
        left_fit = self.best_fit[0]
        right_fit = self.best_fit[1]
        left_lane_indx = ((nonezero_x > (left_fit[0] * (nonezero_y**2) + left_fit[1] * nonezero_y + left_fit[2] - margin)) &
                        (nonezero_x < (left_fit[0] * (nonezero_y**2) + left_fit[1] * nonezero_y + left_fit[2] + margin)))
        right_lane_indx = ((nonezero_x > (right_fit[0] * (nonezero_y**2) + right_fit[1] * nonezero_y + right_fit[2] - margin)) &
                        (nonezero_x < (right_fit[0] * (nonezero_y**2) + right_fit[1] * nonezero_y + right_fit[2] + margin)))

        self.left_x = nonezero_x[left_lane_indx]
        self.left_y = nonezero_y[left_lane_indx]
        self.right_x = nonezero_x[right_lane_indx]
        self.right_y = nonezero_y[right_lane_indx]

        # get fitting
        left_fit = np.polyfit(self.left_y, self.left_x, 2)
        right_fit = np.polyfit(self.right_y, self.right_x, 2)
        # update fitting coefficients
        self.current_fit = [left_fit, right_fit]

        # fitting
        self.left_fit_x = (left_fit[0] * self.ploty ** 2 + left_fit[1] * self.ploty + left_fit[2]).astype(np.int32)
        left_pts = np.vstack((self.left_fit_x, self.ploty)).T.reshape(-1, 1, 2)
        self.right_fit_x = (right_fit[0] * self.ploty ** 2 + right_fit[1] * self.ploty + right_fit[2]).astype(np.int32)
        right_pts = np.vstack((self.right_fit_x, self.ploty)).T.reshape(-1, 1, 2)

        self.out_img = cv2.polylines(self.out_img, [left_pts], False, (0, 0, 255), thickness=2)
        self.out_img = cv2.polylines(self.out_img, [right_pts], False, (0, 0, 255), thickness=2)

        #### visualization ####
        # cv2.imshow('out', self.out_img)
        # cv2.waitKey(0)
        #### visualization ####

        return left_fit, right_fit

    def sanity_check(self):
        cur_left_curvature, cur_right_curvature = self.get_curvature()
        line_dist = self.get_line_distance()
        if cur_left_curvature / cur_right_curvature > 10 or cur_left_curvature / cur_right_curvature < 0.1 or line_dist < 0.2 * self.img_size[0]:
            return False
        else:
            self.cur_curvature = (cur_left_curvature, cur_right_curvature)
            return True

    def update_car_position(self):
        y_eval = self.img_size[1]
        mid_point = self.img_size[0] / 2
        left_bottom_x = self.best_fit[0][0] * (y_eval ** 2) + self.best_fit[0][1] * y_eval + self.best_fit[0][2]
        right_bottom_x = self.best_fit[1][0] * (y_eval ** 2) + self.best_fit[1][1] * y_eval + self.best_fit[1][2]
        self.car_pos = (mid_point - (left_bottom_x + right_bottom_x) / 2) * self.xm_per_pix
        # print('Car position: ', self.car_pos)

    def get_curvature(self):
        y_eval = self.img_size[1]
        left_fit_cur = np.polyfit(self.left_y * self.ym_per_pix, self.left_x * self.xm_per_pix, 2)
        right_fit_cur = np.polyfit(self.right_y * self.ym_per_pix, self.right_x * self.xm_per_pix, 2)
        # compute new curvature in meters
        left_curvature = ((1 + (2 * left_fit_cur[0] * y_eval * self.ym_per_pix + left_fit_cur[1]) ** 2) ** 1.5) \
                         / np.absolute(2 * left_fit_cur[0])
        right_curvature = ((1 + (2 * right_fit_cur[0] * y_eval * self.ym_per_pix + right_fit_cur[1]) ** 2) ** 1.5) \
                          / np.absolute(2 * right_fit_cur[0])
        self.radius_of_curvature = (left_curvature, right_curvature)

        return self.radius_of_curvature

    def get_line_distance(self):
        if self.best_fit is None:
            return self.img_size[0]

        y_eval = np.array([0, self.img_size[1] / 2, self.img_size[1]])
        x_left = self.best_fit[0][0] * (y_eval ** 2) + self.best_fit[0][1] * y_eval + self.best_fit[0][2]
        x_right = self.best_fit[1][0] * (y_eval ** 2) + self.best_fit[1][1] * y_eval + self.best_fit[1][2]
        return np.sum(x_right - x_left) / 3
