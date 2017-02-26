#
# Helpers 
# Helpers module that has all the functions that generate images for the final README
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

# Generate the undistorted version of a distorted image and 
# save the output for reporting purposes
def helper_one():
    mtx = np.loadtxt("model/mtx.dat")
    dist = np.loadtxt("model/dist.dat")
    M = np.loadtxt("model/matrix.dat")
    Minv = np.loadtxt("model/matrix_inv.dat")

    transformer = Transformer()
    transformer.initialise(mtx, dist, M, Minv)

    fig = plt.figure(2, figsize=(9,4))    
    ax1 = fig.add_subplot(121)
    ax1.set_title('Distorted')
    ax2 = fig.add_subplot(122)
    ax2.set_title('Undistorted')
    
    example_img = mpimg.imread('camera_cal/calibration1.jpg')
    undst_example_img = transformer.undistort(example_img)
    ax1.imshow(example_img, cmap='gray')
    ax2.imshow(undst_example_img, cmap='gray')
    plt.tight_layout()
    plt.show()
    fig.savefig("output_images/undistorted_output.png")

def helper_two():
    mtx = np.loadtxt("model/mtx.dat")
    dist = np.loadtxt("model/dist.dat")
    M = np.loadtxt("model/matrix.dat")
    Minv = np.loadtxt("model/matrix_inv.dat")

    transformer = Transformer()
    transformer.initialise(mtx, dist, M, Minv)
    example_img = mpimg.imread('test_images/straight_lines1.jpg')
    warped = transformer.warp(example_img)

    offset = 300
    P1 = [560,  470]
    P2 = [720,  470]
    P3 = [1100, 720]
    P4 = [200,  720]
    src_points = np.float32([P1, P2, P3, P4])
    
    fig = plt.figure(1, figsize=(9,4))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(example_img, cmap='gray')
    final = np.vstack([src_points, src_points[0,:]]) 
    ax1.plot(final[:,0], final[:,1], '-', lw=2, color='r')
    ax1.set_xlim(0, example_img.shape[1])
    ax1.set_ylim(example_img.shape[0],0)
    ax1.set_title("Normal")

    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(warped, cmap='gray')
    iw = example_img.shape[1]
    ih = example_img.shape[0]
    ax2.vlines(offset, 0, iw+10, linestyles='solid', color="r", lw=2)
    ax2.vlines(iw-offset, 0, iw+10, linestyles='solid', color="r", lw=2)    
    ax2.set_title("Warped")    
    ax2.set_xlim(0, warped.shape[1])
    ax2.set_ylim(warped.shape[0],0)
    plt.tight_layout()
    plt.show()    
    fig.savefig("output_images/warped.png")

def helper_three():
    mtx = np.loadtxt("model/mtx.dat")
    dist = np.loadtxt("model/dist.dat")
    M = np.loadtxt("model/matrix.dat")
    Minv = np.loadtxt("model/matrix_inv.dat")

    transformer = Transformer()
    transformer.initialise(mtx, dist, M, Minv)
    image = mpimg.imread('test_images/straight_lines1.jpg')
    warped = transformer.warp(image)

    fig = plt.figure(1, figsize=(9,6))    
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(warped, cmap='gray')
    ax1.set_xlim(0, warped.shape[1])
    ax1.set_ylim(warped.shape[0],0)
    ax1.set_title("Original Warped Image")

    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)         
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx) 
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    grad_binary = np.zeros_like(scaled_sobel)
    thresh_x = (20, 100)
    grad_binary[(scaled_sobel >= thresh_x[0]) & (scaled_sobel <= thresh_x[1])] = 1
    
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(grad_binary, cmap='gray')
    ax2.set_title("Image After Gradient Threshold")    
    ax2.set_xlim(0, grad_binary.shape[1])
    ax2.set_ylim(grad_binary.shape[0],0)

    hsv = cv2.cvtColor(warped, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]
    s_binary = np.zeros_like(s_channel)
    thresh_c = (170, 255)
    s_binary[(s_channel >= thresh_c[0]) & (s_channel <= thresh_c[1])] = 1

    ax3 = fig.add_subplot(2,2,3)
    ax3.imshow(s_binary, cmap='gray')
    ax3.set_title("Image After Color Threshold")
    ax3.set_xlim(0, s_binary.shape[1])
    ax3.set_ylim(s_binary.shape[0],0)

    mask = transformer.color_grad_threshold(warped, sobel_kernel=3, thresh_x=(20, 100),thresh_c=(170, 255))
    ax4 = fig.add_subplot(2,2,4)
    ax4.imshow(mask, cmap='gray')
    ax4.set_title("Image After Combined Threshold")
    ax4.set_xlim(0, mask.shape[1])
    ax4.set_ylim(mask.shape[0],0)

    plt.tight_layout()
    plt.show()
    fig.savefig("output_images/threshold.png")

def helper_four():
    mtx = np.loadtxt("model/mtx.dat")
    dist = np.loadtxt("model/dist.dat")
    M = np.loadtxt("model/matrix.dat")
    Minv = np.loadtxt("model/matrix_inv.dat")

    transformer = Transformer()
    transformer.initialise(mtx, dist, M, Minv)
    image = mpimg.imread('test_images/straight_lines2.jpg')
    undistorted = transformer.undistort(image)
    warped = transformer.warp(undistorted)
    masked = transformer.color_grad_threshold(warped, sobel_kernel=3, thresh_x=(20, 100),thresh_c=(170, 255))

    histogram = np.sum(masked[np.int(masked.shape[0]/2):,:], axis=0)

    # plot histogram
    fig = plt.figure(1, figsize=(9,4))    
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(warped, cmap='gray')
    ax1.set_xlim(0, warped.shape[1])
    ax1.set_ylim(warped.shape[0],0)
    ax1.set_title("Warped Image")  
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(histogram)    
    ax2.set_ylabel("Counts")
    ax2.set_xlabel("Pixel Position")
    ax2.set_aspect('equal')
    ax2.set_title("Histogram")  

    plt.xlim(0,len(histogram))
    plt.ylim(0, warped.shape[0])    
    plt.show()
    fig.savefig("output_images/histogram.png")

def helper_five():
    mtx = np.loadtxt("model/mtx.dat")
    dist = np.loadtxt("model/dist.dat")
    M = np.loadtxt("model/matrix.dat")
    Minv = np.loadtxt("model/matrix_inv.dat")

    transformer = Transformer()
    transformer.initialise(mtx, dist, M, Minv)
    lanefinder = LaneFinder()
    image = mpimg.imread('test_images/straight_lines2.jpg')
    undistorted = transformer.undistort(image)
    warped = transformer.warp(undistorted)
    masked = transformer.color_grad_threshold(warped, sobel_kernel=3, thresh_x=(20, 100),thresh_c=(170, 255))
    left, right = lanefinder.find_peaks(masked)
    out_img, left_fitx, right_fitx, ploty = lanefinder.sliding_window(masked, left, right)
    
    fig = plt.figure(1, figsize=(9,4))    
    
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(warped, cmap='gray')
    ax1.set_xlim(0, warped.shape[1])
    ax1.set_ylim(warped.shape[0],0)
    ax1.set_title("Thresholded Image")  
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(out_img, cmap='gray')
    ax2.plot(left_fitx, ploty, color='yellow')
    ax2.plot(right_fitx, ploty, color='yellow')
    ax2.set_xlim(0, out_img.shape[1])
    ax2.set_ylim(out_img.shape[0],0)
    ax2.set_title("Detected Lines")  
    plt.show()    
    fig.savefig("output_images/detected_lines.png")

if __name__ == '__main__':

    helper_five()