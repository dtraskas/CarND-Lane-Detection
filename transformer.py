#
# Lane Detection 
# Transformer module takes an image and applies a perspective transform
#
# Dimitrios Traskas
#
#
import numpy as np
import cv2
import matplotlib.image as mpimg

class Transformer:
    
    def __init__(self, offset, src_points, mtx, dist):
        self.offset = offset
        self.src_points = src_points        
        self.mtx = mtx
        self.dist = dist

    def unwarp(self, image):
        
        # Use the OpenCV undistort() function to remove distortion
        undist = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

        # Grab the image shape
        img_size = (undist.shape[1], undist.shape[0])                
        dst_points = np.float32([[self.offset, self.offset], 
                                 [img_size[0] - self.offset, self.offset], 
                                 [img_size[0] - self.offset, img_size[1]], 
                                 [self.offset, img_size[1]]])
        
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(self.src_points, dst_points)
        np.savetxt("model/matrix.dat", M)

        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

        return warped, M
    
    # Calculates directional gradient
    def abs_sobel_thresh(self, image, orient='x', sobel_kernel=3, thresh=(0, 255)):
        
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
    def mag_thresh(self, image, sobel_kernel=3, mag_thresh=(0, 255)):
        
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
    def dir_threshold(self, image, sobel_kernel=3, thresh=(0, np.pi/2)):       

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        
        # Take the absolute value of the gradient direction, 
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        dir_binary = np.zeros_like(absgraddir)
        dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        return dir_binary