##########################################################################
#                                                                        #
#  © 2026 Regents of the University of Minnesota. All rights reserved.   #
# This program is shared under the terms and conditions of the GNU       #
# Affero General Public License 3.0, License. Further details about      #
# the GNU Affero GPL 3.0 license are available in the LICENSE text file. #
#                                                                        #
##########################################################################

import numpy as np
import cv2
import inspect
import math
from numpy.linalg import inv

# ////////////////////////////////////////////////////////////////////////////////////
# //                                                                                //
# //  class TM_trajectory                                                           //
# //                                                                                //
# ////////////////////////////////////////////////////////////////////////////////////

# OpenCV fisheye model is strickly radial distortion coefficients:
#     r_d = r*(1 + k1*r^2 + k2*r^4 + k3*r^6 + k4*r^8)
# and for PnP solution with n >= 4 coplanar points, the openCV distortion input vector would be
# (according to https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html, and 
#  https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d)
#  distCoeffs_IN = [k1,k2, p1=0, p2=0, k3, k4, k5=0, k6=0]
class TM_trajectory:
    valid=True
    Rmtx = np.eye(3)
    scalefac = 1.0
    H_w2c = np.zeros([3,3])   # 2D homography: X,Y,Z=0 world  ->  x,y camera img
    H_c2w = np.zeros([3,3])   # 2D homography: x,y camera img ->  world X,Y,Z=0

    Tvec = np.array([[0,0,0]], dtype=np.float64)
    Kmtx = np.zeros([3,3])
    extrinsic = np.zeros([3,4], dtype=np.float64)
    def __init__(self, XYZworld_pnts, xyimage_pnts, K: np.ndarray, D=0, sf_IN=1.0):

        if len(K.shape) != 2:
            print(f"{inspect.currentframe().f_code.co_name}: Intrinsic matrix input, must be a 2-dim nparray!")
            self.valid=False
            return
        
        if len(XYZworld_pnts) == 0 or len(xyimage_pnts)==0:
            self.valid=False
            return
        
        # check if image data are assumed to originate from corrected image or the original, distorted one.
        if isinstance(D,np.ndarray):
            D_in = np.zeros([8,1], dtype=np.float64)
            D_in[0] =D[0]
            D_in[1] =D[1]
            D_in[4] =D[2]
            D_in[5] =D[3]
        else:
            D_in=0

        # flags = cv2.SOLVEPNP_ITERATIVE | cv2.SOLVEPNP_IPPE
        # rvec_init = cv2.Rodrigues(self.Rmtx)
        # tvec_init = np.array([[0.0,0.0,35.0]])
        # return World w.r.t. camera transformation
        if max( XYZworld_pnts.shape ) < 4:
            print(f"{inspect.currentframe().f_code.co_name} ERR: XYZ points < 4!")
            self.valid=False
            return
        
        if XYZworld_pnts.shape[0] < XYZworld_pnts.shape[1]:
            XYZworld_pnts = XYZworld_pnts.T
        if  XYZworld_pnts.shape[1] == 2: # assume Zw=0.0
            XYZworld_pnts = np.hstack( XYZworld_pnts, np.zeros(XYZworld_pnts.shape[0],1))

        self.PnPscalefac = sf_IN
        xyimage_pnts = xyimage_pnts * sf_IN
        intxyi = np.round(xyimage_pnts)
        print(f"...solving PnP using {len(intxyi)} point pairs...")
        pnpflags = cv2.SOLVEPNP_ITERATIVE
        pnpflags = cv2.SOLVEPNP_IPPE
        print(f"XYZworld_pnts.shape={XYZworld_pnts.shape}, intxyi.shape={xyimage_pnts.shape}")
        retval, rvec, tvec = cv2.solvePnP(XYZworld_pnts, xyimage_pnts,  K, D_in, flags=pnpflags)
        print(f"{inspect.currentframe().f_code.co_name} PnP result status={retval} ")
        self.Rmtx = cv2.Rodrigues(rvec)
        self.Tvec = tvec.reshape(3,1)
        self.Kmtx = K
        
        self.extrinsic[0:3,0:3] = self.Rmtx[0]
        print(self.Tvec, self.Tvec.shape)
        
        self.extrinsic[0:3,3] = self.Tvec[:,0]
        self.Mw2c = K @ self.extrinsic  # (3x3) X (3x4)
        # this is planar 2D homography going from world X,Y(Z=0) to 2D img (x,y,1)
        self.H_w2c[0:3,0:2] = self.Mw2c[0:3,0:2]
        self.H_w2c[0:3,2] = self.Mw2c[0:3,3]
        #  planar 2D inverse homography:  img (u,v,1) to (X,Y,Z=0,1)
        self.H_c2w = inv(self.H_w2c)
        return

    def setScaleFactor(self, scale_in):
        if scale_in > 0:
            self.scalefac=scale_in
        else:
            f"{inspect.currentframe().f_code.co_name} WARN: input scale factor, {scale_in} < 0!"
        return
    
    # for Z=0
    # will rescale pixels with pixel/img scaling factor.
    # scale factor < 1 will increase pixel values to the larger "original" image
    # assumed for the PnP solution.
    # scale factor > 1 will decrease pixel values to a smaller image
    def pixel2worldXY_homog(self, xpix, ypix, H_pix2w=None):
        if not self.valid:
            return -9999.9999,-9999.9999
        
        sf = 1.0/self.scalefac
        xpix = sf*xpix
        ypix = sf*ypix
        xyi_vec = np.asarray([[xpix,ypix,1.0]]).reshape(3,1)
        if H_pix2w==None:
            XYw = self.H_c2w @ xyi_vec
        else:
            XYw = H_pix2w @ xyi_vec
        return XYw[0]/XYw[2], XYw[1]/XYw[2]

    # it uses a 3x3 homography if Zw=0
    # scalefac < 1 will reduce pixel values to target image size.
    def world2pixel(self,Xw, Yw, Zw, returnint=False, H_w2pix=None):
        if not self.valid:
            return -1,-1
    
        s = self.scalefac
        if Zw==0:
            XYZvec =  np.array([[Xw,Yw,1]]).reshape(3,1)
            if H_w2pix is None:
                xyp = self.H_w2c.dot( XYZvec )
            else:
                xyp = H_w2pix @ XYZvec
        else: 
            #print(f"world2pixel: Xw,Yw,Zw={Xw},{Yw},{Zw}")
            XYZvec = np.array([[Xw,Yw,Zw,1]]).reshape(4,1)
            if H_w2pix is None:
                xyp = self.Mw2c @ XYZvec
            else:
                xyp = H_w2pix @ XYZvec

        if not returnint:
            return  s*xyp[0]/xyp[2], s*xyp[1]/xyp[2]
        else:
            xscalar = s*xyp[0].item()
            yscalar = s*xyp[1].item() 
            zscalar = xyp[2].item()
            return math.ceil(xscalar/zscalar), math.ceil( yscalar/zscalar)

