
#
# Pipeline
# The image processing pipeline that generates the final video
#
# Dimitrios Traskas
#
#

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

from calibrator import CameraCalibrator
from transformer import Transformer 
from lanefinder import LaneFinder

# Definition of global transformer used in the process image 
# function because a function parameter cannot be passed
globalTransformer = Transformer()
def process_image(image):
    
    lanefinder = LaneFinder()        
    undistorted = globalTransformer.undistort(image)
    warped = globalTransformer.warp(undistorted)

    masked = globalTransformer.color_grad_threshold(warped, sobel_kernel=3, thresh_x=(20, 100),thresh_c=(170, 255))
    left, right = lanefinder.find_peaks(masked)
    out_img, left_fitx, right_fitx, ploty = lanefinder.sliding_window(masked, left, right)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(masked).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, globalTransformer.get_minv(), (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

    return result    

# Process the entire video
def process_video(inp_fname, out_fname):    
    
    mtx = np.loadtxt("model/mtx.dat")
    dist = np.loadtxt("model/dist.dat")
    M = np.loadtxt("model/matrix.dat")
    Minv = np.loadtxt("model/matrix_inv.dat")

    globalTransformer.initialise(mtx, dist, M, Minv)
    
    clip = VideoFileClip(inp_fname)    
    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(out_fname, audio=False)

if __name__ == '__main__':
    
    phase = 'process'
    if phase == 'calibration':
        corners = (9, 6)
                
        calibrator = CameraCalibrator('camera_cal', corners)
        mtx, dist = calibrator.calibrate()    
        
        offset = 300
        P1 = [560,  470]
        P2 = [720,  470]
        P3 = [1100, 720]
        P4 = [200,  720]
        src_points = np.float32([P1, P2, P3, P4])

        cal_image = mpimg.imread('test_images/test1.jpg')
        calibrator.calc_perspective(cal_image, mtx, dist, offset, src_points)
        
        plt.figure()    
        plt.imshow(cal_image, cmap='gray')
        plt.show()
    
    if phase == 'transform':
        image = mpimg.imread('test_images/test2.jpg')

        mtx = np.loadtxt("model/mtx.dat")
        dist = np.loadtxt("model/dist.dat")
        M = np.loadtxt("model/matrix.dat")
        Minv = np.loadtxt("model/matrix_inv.dat")

        transformer = Transformer()
        transformer.initialise(mtx, dist, M, Minv)        
        lanefinder = LaneFinder()
        
        undistorted = transformer.undistort(image)
        warped = transformer.warp(undistorted)

        masked = transformer.color_grad_threshold(warped, sobel_kernel=3, thresh_x=(20, 100),thresh_c=(170, 255))
        left, right = lanefinder.find_peaks(masked)
        out_img, left_fitx, right_fitx, ploty = lanefinder.sliding_window(masked, left, right)
        
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(masked).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        curve_l = lanefinder.get_curvature(left_fitx, masked)
        print(curve_l)

        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
        plt.figure()
        plt.imshow(result)
        plt.show()

    if phase == 'process':
        print("Started processing video...")
        process_video('project_video.mp4', 'project_output.mp4')
        print("Completed video processing!")