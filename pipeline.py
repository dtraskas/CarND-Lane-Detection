
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

# Definition of global Transformer and global LaneFinder used in the process image function
globalTransformer = Transformer()
globalLaneFinder = LaneFinder()
def process_image(image):
    
    undistorted = globalTransformer.undistort(image)
    warped = globalTransformer.warp(undistorted)

    masked = globalTransformer.color_grad_threshold(warped, sobel_kernel=9, thresh_x=(20, 100),thresh_c=(60, 255))
    left, right = globalLaneFinder.find_peaks(masked)
    left_fit, right_fit, leftx, lefty, rightx, righty = globalLaneFinder.sliding_window(masked, left, right)
    final_result = globalLaneFinder.get_lane(undistorted, masked, left_fit, right_fit)
    left_r, right_r, offset = globalLaneFinder.get_curvature(masked, left_fit, right_fit)    
    final_result = globalLaneFinder.add_stats(final_result, left_r, right_r, offset)

    return final_result    

# Process the entire video
def process_video(inp_fname, out_fname):    
    
    mtx = np.loadtxt("model/mtx.dat")
    dist = np.loadtxt("model/dist.dat")
    M = np.loadtxt("model/matrix.dat")
    Minv = np.loadtxt("model/matrix_inv.dat")

    globalTransformer.initialise(mtx, dist, M, Minv)
    globalLaneFinder.initialise(Minv)

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

    if phase == 'process':
        print("Started processing video...")
        process_video('project_video.mp4', 'project_output.mp4')
        print("Completed video processing!")