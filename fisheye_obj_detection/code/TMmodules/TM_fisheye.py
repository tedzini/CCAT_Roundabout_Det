##########################################################################
#                                                                        #
#  © 2026 Regents of the University of Minnesota. All rights reserved.   #
# This program is shared under the terms and conditions of the GNU       #
# Affero General Public License 3.0, License. Further details about      #
# the GNU Affero GPL 3.0 license are available in the LICENSE text file. #
#                                                                        #
##########################################################################

import cv2
import numpy as np
import json
import inspect
import math

try:
    import vpi
    vpi_found = True
except ModuleNotFoundError:
    vpi_found = False
    print("module 'vpi' not found!")

# ////////////////////////////////////////////////////////////////////////////////////
# //                                                                                //
# //  class TM_fisheye                                                              //
# //                                                                                //
# ////////////////////////////////////////////////////////////////////////////////////
class TM_fisheye:
    map1=None
    map2=None
    balance=1.0
    h=0
    w=0
    vpifound = vpi_found
    
    def load_calibration(self, filename):
        with open(filename, 'r') as f:
            calibration_data = json.load(f)
            self.K = np.array(calibration_data['K'])
            print(f"K => {self.K}")
            self.D = np.array(calibration_data['D'])
            self.h = calibration_data['imgrows']
            self.w = calibration_data['imgcols']
            self.focal = calibration_data['focal']
            self.sensorW = calibration_data['sensorW']
            self.sensorH = self.sensorW # note: fisheye camera sensor and image are square!
            self.util_UndistortMap()

    def resize(self, new_imgsize: int ):
        self.scalefac = new_imgsize/self.w
        self.focal *= self.scalefac
        self.w = new_imgsize
        self.h = new_imgsize
        self.K = self.scalefac * self.K
        self.K[2,2] = 1.0
        print(f"resize(): K => {self.K}")
        self.util_UndistortMap()
        
    def util_UndistortMap(self):
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.K, self.D, (self.w, self.h), np.eye(3),1)
        self.Knew=new_K
        print(f"\nnew_K => {new_K}")
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), new_K, (self.w, self.h), cv2.CV_16SC2)
        
        # set up VPI warp grid for Jetson
        if self.vpifound is True:
            print(f"shape of distortion coeffs: {self.D.shape}")
            self.grid = vpi.WarpGrid((self.w, self.h))
            self.warp = vpi.WarpMap.fisheye_correction(self.grid, K=self.K[:2,:3], X=np.eye(3,4),
                                mapping=vpi.FisheyeMapping.STEREOGRAPHIC,
                                coeffs=self.D.flatten().tolist())
            print(f"VPI warp grid created with size {self.warp.grid.size}")
        else:
            print("VPI not found, skipping warp grid creation...")
            self.warp = None
            self.grid = None

    def getKD(self):
        return self.K, self.D
    
    def getUpdatedK(self):
        return self.Knew
    
    def undistort_image(self, image, balance=1):
        if self.vpifound is True:
            return self.undistort_img_vpi(image)
        else:
            return self.undistort_img(image, balance)

    def rescale_image(self, image, target_size : tuple):
        if self.vpifound is True:
            return self.rescale_img_vpi(image, target_size)
        else:
            return self.rescale_img(image, target_size)
        
    def rescale_img(self, image, targetsz : tuple ):
        resized_img = cv2.resize( image, targetsz)
        return(resized_img)
    
    def undistort_img(self, image, balance=1):
        if self.h  != image.shape[0] or self.w != image.shape[1]:
            print(f"{inspect.currentframe().f_code.co_name}: img size ({image.shape[:2]} != calibration img size {self.h}x{self.w}")
            return image
        if self.map1 is None or self.map2 is None:
            print(f"{inspect.currentframe().f_code.co_name}: calibration maps not set!")
            return image
        undistorted = cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted

    def rescale_img_vpi(self, image, targetsz : tuple):
        # convert to VPI image
        #  if self.h  != image.shape[0] or self.w != image.shape[1]:
        #     print(f"{inspect.currentframe().f_code.co_name}: img size ({image.shape[:2]} != calibration img size {self.h}x{self.w}")
        #     return image
        with vpi.Backend.CUDA: # vpi.Backend.PVA, vpi.Backend.CUDA, vpi.Backend.CPU
            vpi_image = vpi.asimage(image, vpi.Format.BGR8)
            resized_vpi = vpi_image.rescale( targetsz, interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
            resized_img = resized_vpi.cpu()
            return resized_img
 
       
    def undistort_img_vpi(self, image):
 
        # only CPU and CUDA backends are supported with remapping
        with vpi.Backend.CUDA: # vpi.Backend.PVA, vpi.Backend.CUDA, vpi.Backend.CPU
            # convert color image to VPI image
            vpi_image = vpi.asimage(image, vpi.Format.BGR8)
            
            output = vpi_image.remap(self.warp, interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO)
            undistorted = output.cpu()
#        with output.rlock(vpi.MemType.CPU) as data:
#             undistorted = data

        # If necessary, convert color space (example: NV12 to BGR)
        # cv_mat_bgr = cv2.cvtColor(cv_mat, cv2.COLOR_YUV2BGR_NV12)
        return undistorted