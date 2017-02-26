#
# Lane Detection 
# Calibrator module that calibrates the camera based on a set of chessboard images provided
#
# Dimitrios Traskas
#
#
import numpy as np
import cv2
import matplotlib.image as mpimg
import os

class CameraCalibrator:

    def __init__(self, path, corners):
        self.path = path
        self.corners = corners
    
    def read_filenames(self):  
        filecount = len([f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))])      
        images = np.empty(filecount, dtype='object')     
        cnt = 0
        for filename in os.listdir(self.path):
            fullFilename = os.path.join(self.path,filename)        
            
            images[cnt] = fullFilename
            cnt += 1    
        
        return images
    
    def calibrate(self):        
        
        objpoints = []
        imgpoints = []
        
        objp = np.zeros((self.corners[0] * self.corners[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.corners[0], 0:self.corners[1]].T.reshape(-1,2)
                
        filenames = self.read_filenames()
        images = []
        for fname in filenames:
            image = mpimg.imread(fname)            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ret, new_corners = cv2.findChessboardCorners(gray, self.corners, None)            
            if ret == True:
                imgpoints.append(new_corners)    
                objpoints.append(objp)
                
                img = cv2.drawChessboardCorners(image, self.corners, new_corners, ret)                
                images.append(img)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        print("Saving camera calibration coefficients...")
        np.savetxt("model/mtx.dat", mtx)
        np.savetxt("model/dist.dat", dist)        
        return mtx, dist

    def calc_perspective(self, image, mtx, dist, offset, src_points):
        
        undist = cv2.undistort(image, mtx, dist, None, mtx)
        # Grab the image shape
        img_size = (undist.shape[1], undist.shape[0])                
        dst_points = np.float32([[offset, offset], 
                                 [img_size[0] - offset, offset], 
                                 [img_size[0] - offset, img_size[1]], 
                                 [offset, img_size[1]]])
        
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        np.savetxt("model/matrix.dat", M)

        Minv = cv2.getPerspectiveTransform(dst_points, src_points)
        np.savetxt("model/matrix_inv.dat", Minv)