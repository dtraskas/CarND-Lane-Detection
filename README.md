**Advanced Lane Finding Project**

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
[image2]: ./output_images/warped.jpg "Undistorted and Warped"


[image3]: ./examples/warped_straight_lines.jpg "Warp Example"
[image4]: ./examples/color_fit_lines.jpg "Fit Visual"
[image5]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

###Camera Calibration

The code for this step is contained in the `calibrator.py` file and more specifically the calibrate function. The calculated calibration matrix and distortion coefficients are saved during a one-off calibration processing step in the `/model` folder. Calibration takes place by utilising all the chessboard images within the `/camera_cal` folder. The images are loaded and a list of coordinates of the chessoard corners is prepared. The assumption made here is that the chessboard is fixed on the (x, y) plane at z=0. The `objp` list is an array of coordinates that gets appended to `objpoints` every time a successful detection of all the chessboard corners in a test image is made. Just to note here that the chessboard corners for this calibration are of size (9,6). 

The output `objpoints`, `imgpoints` are then used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. The `transformer.py` file contains the `transformer` class that essentially takes as parameters in its constructor the distortion coefficients and calibration matrix and provides a number of transformation functions such as `undistort()`. 

One more step of the calibration process as an one-off is the calculation of the perspective transform matrix and its inverse. This is computed in the `calibrator.py` class and the `calc_perspective()` function. Once these matrices are calculated they are also saved within the `model` folder.

###Pipeline (single images)

####1. Distortion Correction
Using the `undistort` function mentioned earlier and the already saved matrices we can produce an undistorted image as can be seen below:

![alt text][image1]

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

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]




####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

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

