#%%

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from calibrator import CameraCalibrator

# Calculates directional gradient
def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)    
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))

    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))        
    
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))    
    # Create a mask of 1's where the scaled gradient magnitude 
    # is > thresh_min and < thresh_max
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary


# Calculates gradient magnitude
def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return mag_binary

# Calculate gradient direction
def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):       

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return dir_binary

def test():
    image = mpimg.imread('images/signs_vehicles_xygrad.png')    

    # Choose a Sobel kernel size
    ksize = 11 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 120))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 120))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(50, 150))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.5, 1.3))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # Plot the result
    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(5, 5))
    ax1.imshow(image)
    ax2.set_title('Original Image', fontsize=10)
    ax2.imshow(combined, cmap='gray')
    ax2.set_title('Thresholded Gradient', fontsize=10)
    plt.show()


if __name__ == '__main__':
    calibrator = CameraCalibrator('camera_cal', (9, 6))
    mtx, dist, images = calibrator.calibrate()    
    image = cv2.undistort(images[0], mtx, dist, None, mtx)
    plt.figure()
    
    plt.imshow(images[0])
    plt.show()
