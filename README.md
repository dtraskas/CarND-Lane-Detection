# Advanced Lane Finding

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_output.png "Undistorted"
[image2]: ./output_images/warped.png "Undistorted and Warped"

[image3]: ./output_images/threshold.png "Thresholds Applied"
[image4]: ./output_images/histogram.png "Histogram"
[image5]: ./output_images/detected_lines.png "Detected Lines"

[image6]: ./output_images/undistorted_test.png "Undistorted Test"

[image7]: ./output_images/example_output.png "Output"
[image8]: ./output_images/problem.png "Output"
[video1]: ./project_video.mp4 "Video"

###Camera Calibration

The code for this step is contained in the `calibrator.py` file and more specifically the calibrate function. The calculated calibration matrix and distortion coefficients are saved during a one-off calibration processing step in the `/model` folder. Calibration takes place by utilising all the chessboard images within the `/camera_cal` folder. The images are loaded and a list of coordinates of the chessoard corners is prepared. The assumption made here is that the chessboard is fixed on the (x, y) plane at z=0. The `objp` list is an array of coordinates that gets appended to `objpoints` every time a successful detection of all the chessboard corners in a test image is made. Just to note here that the chessboard corners for this calibration are of size (9,6). 

The output `objpoints`, `imgpoints` are then used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. The `transformer.py` file contains the `transformer` class that essentially takes as parameters in its constructor the distortion coefficients and calibration matrix and provides a number of transformation functions such as `undistort()`. 

One more step of the calibration process as an one-off is the calculation of the perspective transform matrix and its inverse. This is computed in the `calibrator.py` class and the `calc_perspective()` function. Once these matrices are calculated they are also saved within the `model` folder. To demonstrate that the calibration took place is correct we plot below the result of the `undistort()` function.

![alt text][image1]

###Pipeline (single images)

####1. Distortion Correction

Using the `undistort()` function mentioned earlier and the already saved matrices we can produce an undistorted image as can be seen below:

![alt text][image6]

####2. Perspective Transform 

The code for perspective transform resides within the `transformer.py` class and includes a function called `warp()`. The `warp()` function takes as inputs an image (`image`) and utilises the already calculated perspective transform matrix. This matrix and its inverse are loaded during initialisation time in the constructor. The inverse matrix is used to unwarp images as will be seen at a later stage of the pipeline.

In order to calculate the perspective transform matrix I chose the source and destination points by looking at one of the test images. The code for this calculation resides within the `pipeline.py` module and is within the calibration phase. As you can also see I have chosen an offset of `300` and the following source points and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 560, 470      | 300, 300      | 
| 720, 470      | 980, 300      |
| 1100, 720     | 980, 720      |
| 200, 720      | 300, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image2]

####3. Thresholds

The color and gradient thresholds were applied on the test images to generate a binary image. For the color threshold the saturation channel is first separated from the RGB image and subsequently thresholded using a tuple of values. For the gradient threshold the image is converted to grayscale, the x gradient is then calculated using the cv2.Sobel(), the absolute values are subsequently scaled and thresholds applied using a tuple of values. Once the two thresholds are calculated they are combined as can be seen here: 

`combined[(s_binary == 1) | (grad_binary == 1)] = 1`

The function that combines the two thresholds can be found in the `transformer.py` and it's `color_grad_threshold()`.

Below you can see an example of thresholds applied using one of the test images provided:

![alt text][image3]

####4. Line Detection

The line detection function `find_peaks()` can be found in the LaneFinder class in `lanefinder.py`. The approach is to first take an image, split it horizontally and take the bottom half of that. Using a histogram to find the peaks where pixels have higher intensity we can detect the lines as can be seen below:

![alt text][image4]

Then we define a number of sliding windows and loop through them in order to identify the borders of the windows, find the non-zero pixels and save those in a left and right list that we use at a later stage when we fit a polynomial function that results to this final image:

![alt text][image5]

The function described can be found in `lanefinder.py` and `sliding_window()`.

####5. Line Curvature

In the `lanefinder.py` function and the `get_curvature()` function I calculate the curvature using the formula:

R_curve = (1 + (2Ay+B)<sup>2</sup>)<sup>3/2</sup> / |2A|

I then convert the outcome from pixels to meters for the real world space. An offset is calculated by assuming the camera is mounted at the center of the car, such that the lane center is the midpoint at the bottom of the image between the two lines detected. The `get_offset()` function returns the calculated offset for every image processed in the pipeline.

####6. Final Result

The final result once I plot back to the original image the lane can be seen below:

![alt text][image7]

In the `lanefinder.py` class and in the `get_lane()` function I plot back the final lane that essentially is a polygon on the warped image and unwarped by using the inverse perspective transform matrix.

---

###Pipeline (video)

Here's a [link to my video result](./project_output.mp4). The video was generated by using the `moviepy.editor` package and `VideoFileClip` set of functions, as well as my newly defined classes. The `process_video()` function is used to open a video file and pass `process_image()` as a parameter to the `clip.fl_image()` function that processes the array of frames found in the specified video. A global transformer is initialised in order to be passed into the `process_image()` function.

---

###Discussion

I spent a lot of my time in this project finetuning the parameters of the threshold functions in order to mask the line pixels. Shadows, changes in lighting and other objects on the road seem to heavily affect the thresholding process.

It would be useful for a future extension of this work to detect areas of pixels that get discarded. This would require to split the image in areas where I use different thresholding techniques and masking. Additionally object detection using a separate algorithm could exclude areas of the image that are not relevant but affect the lane detection.

I would also like to spend a bit more time adding a more efficient search function for the next frame based on the previous detections. That would result in a more robust result and faster processing especially in real-time. 

My biggest concern however with this entire approach is the applicability of the lane detection algorithm in roads where lanes are not clearly defined or even absent from the road. In many areas across the world lanes are non-existent or when they exist only hints of them can be seen. 