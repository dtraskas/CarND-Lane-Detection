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

`R_curve = (1 + (2Ay+B)<sup>2</sup>)<sup>3/2</sup> / |2A|`

####6. Final Result

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

