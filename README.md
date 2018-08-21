## Advanced Image Processing on Lane Detection

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[image1]: ./output_images/distortion_comparison.jpg "Distortion Comparison"
[image2]: ./output_images/straight_lines2.jpg "Raw Image"
[image3]: ./output_images/undist_straight_lines2_line.jpg "Undistortion Image"
[image4]: ./output_images/bin_straight_lines2.jpg "Thresholded"
[image5]: ./output_images/perepective_region.jpg "Perspective Region"
[Image6]: ./output_images/perepective_region_warped.jpg "Perspective Region Warped"
[image7]: ./output_images/lane_finding_sliding_window.png "Sliding Window"
[Image8]: ./output_images/lane_finding_tracking.png "Lane Tracking"
[Image9]: ./output_images/overlay.png "Overlay"

### Camera Calibration

The code for this step is in the file called `camera_calibration.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. A distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

The above image is the raw image. By drawing a straight line along with the road lanes, we can easily see that the road lane is not straight, which means it is distorted.

![alt text][image3]

The image above is distortion-corrected image from the same raw image. This time, we can see the road lane perpeftly matches the straight line I drew, which means the distortion coefficients are good enough to remove the image distortion caused by the camera lens.

#### 2. Use color transforms, gradients or other methods to create a thresholded binary image.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 5 through 16, 42 through 48 and 57 through 57 in `image_processing.py`).  Here's an example of my output for this step.  

![alt text][image4]

#### 3. Perform a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transfrom()`, which appears in lines 69 through 77 in the file `image_processing.py`.  The `perspective_transfrom()` function takes as inputs an image (`img`), as well as source (`src_points`) and destination (`dst_points`) points.  I chose the hardcode the source and destination points in the following manner:

```python
scaled_size = 1
image_size = (int(1280 * scaled_size), int(720 * scaled_size))
offset = image_size[1] * 0.35
perspective_src_points = scaled_size * np.float32(
    [[233, 694], 
     [595, 450], 
     [686, 450], 
     [1073, 694]])
perspective_dst_points = np.float32(
    [[offset, image_size[1]], 
     [offset, 0],
     [image_size[0] - offset, 0], 
     [image_size[0] - offset, image_size[1]]])
```

This resulted in the following source and destination points:

|  Source   | Destination |
| :-------: | :---------: |
| 233, 694  |  252, 720   |
| 595, 450  |   252, 0    |
| 686, 450  |   1028, 0   |
| 1073, 694 |  1028, 720  |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

![alt text][image6]

#### 4. Identify lane-line pixels and fit their positions with a polynomial.

I did this in the class Line() in `line.py`

There are two modes for finding the lanes. Blind search(code line from 62 through 141) and Tracking(code line from 143 through 184). For previous frame, if there was no satidfying lanes detected(the lane found cound not pass the sanity check, which is in the code line from 186 through 192), I do a blind search. Otherwise, I do tracking based on the previous lane detection.

![alt text][image7]

The above image shows the blind search method. I seached all the nonzero points based on the warped binary image then made a histogram which counts the number of nonzero points on each column of the lower half of the image. Within the histogram, I take the two peak column indices from the left and right side. With the preset window size, I searched all the points within that window and collect the x and y coordinates. Once a window has finished, I kept sliding the window upwards to keep searching and if I get enough points, I just recenter the window for the next time's searching. Alter I collect all of the points, I fit the points to a second order polynomial for both left and right points. Then I have the coefficients of the two polynomial and am able to get the fit line, which is drawed with red color.

![alt text][image8]

The above image shows the tracking lanes seaching. If the lanes are detected in the last frame, I do not need to do the blind search again for saving both computation power as well as time. Based to the previous points' coordinated, I searched all of the nonzero points within a margin and collect them. Using the same fitting method I will be able to obtain the polynomial coefficients and draw the fit lanes with red color showed above.

#### 5. Calculate the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the class Line() in `line.py`

For getting the curvatue, the code is from line 203 through 214. I first transformed the lane coordinated in pixel unit to meter unit using the general rate. Then I fit the lines using a second order polynomial. After that I have the coefficients for the polynomial and using y=image height to compute the curvature at the bottom of the image using the curvature expression function.

For getting the car position, the code is from line 195 through 200. Using the coefficients from the fit lane polynomial in pixel unit, I am able to get the bottom x position of the two lanes in the image. The mid point of those two points will be the road lane center. Also we know the image center is the vehicle's position so I am able to compute the offset. By multiplying the pixel to meter ratio I will be able to know how much the car is off the road center, either left or right.	

#### 6. Result.

I implemented this step in lines 54 through 57 in my code in `mainm.py` and in the function `get_warpback_overlay_img()` in lines 80 through 103 in and lines from 106 through 128 for `add_img_info()` in code `image_processing.py`.  Here is an example of my result on a test image:

![alt text][image9]

---

### Video

Here's a [link to my video result](https://youtu.be/rT0CF0HcxlU)
