import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from calibrator import CameraCalibrator
from transformer import Transformer 

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
    
    phase = 'transform'
    
    if phase == 'calibration':
        corners = (9, 6)
        calibrator = CameraCalibrator('camera_cal', corners)
        calibrator.calibrate()    
    
    test_image = 'test_images/straight_lines1.jpg'
    if phase == 'transform':
        mtx = np.loadtxt("model/mtx.dat")
        dist = np.loadtxt("model/dist.dat")
        image = mpimg.imread(test_image)
        
        offset = 300
        P1 = [560,  470]
        P2 = [720,  470]
        P3 = [1100, 720]
        P4 = [200,  720]
        src_points = np.float32([P1, P2, P3, P4])
        transformer = Transformer(offset, src_points, mtx, dist)
        top_down, perspective_M = transformer.unwarp(image)

    plt.figure()    
    plt.imshow(top_down)
    plt.show()