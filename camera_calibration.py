import platform
import glob
import numpy as np
import cv2


class CameraCalibration():
    def __init__(self, images_path):
        self.intrinsic_mat = None
        self.dist_paras = None
        self.image_paths = glob.glob(images_path + '*.jpg')

    def calibrate(self):
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        object_points = []
        img_points = []

        for idx, img_path in enumerate(self.image_paths):
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # find corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            if ret is True:
                object_points.append(objp)
                img_points.append(corners)

                cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                if platform.system() == 'Windows':
                    write_name = './output_images/draw_' + img_path[img_path.rfind('\\') + 1:]
                else:
                    write_name = './output_images/draw_' + img_path[img_path.rfind('/') + 1:]
                cv2.imwrite(write_name, img)

        #         calibrate
        test_img = cv2.imread('./camera_cal/calibration1.jpg')
        img_size = (test_img.shape[1], test_img.shape[0])
        ret, self.intrinsic_mat, self.dist_paras, rvecs, tves = cv2.calibrateCamera(object_points, img_points, img_size,
                                                                                    None, None)

        # Generate undistorted images
        # dst = cv2.undistort(test_img, self.intrinsic_mat, self.dist_paras, None, self.intrinsic_mat)
        # raw_small = cv2.resize(test_img, (0, 0), fx=0.25, fy=0.25)
        # cv2.putText(raw_small, 'Raw', (raw_small.shape[1] // 2 - 50, raw_small.shape[0] // 2),
        # 			fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255))
        # dst_small = cv2.resize(dst, (0, 0), fx=0.25, fy=0.25)
        # cv2.putText(dst_small, 'Undistorted', (dst_small.shape[1] // 2 - 120, dst_small.shape[0] // 2),
        # 			fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.3, color=(0, 0, 255))
        # distortion_comparison = np.hstack([raw_small, dst_small])
        # cv2.imshow('distortion comparison', distortion_comparison)
        # cv2.imwrite('./output_images/distortion_comparison.jpg', distortion_comparison)

        print('Camera calibration finished! Press any key to continue.')

    def undistort_images(self):
        print('Images distortion on sample imges started...')
        for img_path in self.image_paths:
            img = cv2.imread(img_path)
            undist = cv2.undistort(img, self.intrinsic_mat, self.dist_paras, None, self.intrinsic_mat)
        # if platform.system() == 'Windows':
        # 	cv2.imwrite('./output_images/undist_' + img_path[img_path.rfind('\\') + 1:], undist)
        # else:
        # 	cv2.imwrite('./output_images/undist_' + img_path[img_path.rfind('/') + 1:], undist)
        print('Images distortion correction on sample imagesfinished!')

    def distort_correction(self, img):
        undist = cv2.undistort(img, self.intrinsic_mat, self.dist_paras, None, self.intrinsic_mat)

        return undist


    def get_intrinsic(self):
        return self.intrinsic_mat

    def get_distortion_paras(self):
        return self.dist_paras


if __name__ == '__main__':
    calibration = CameraCalibration('camera_cal/')
    calibration.calibrate()
