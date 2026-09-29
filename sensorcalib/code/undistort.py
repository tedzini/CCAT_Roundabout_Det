#!/usr/bin/env python3
###############################################################################
# Inspired by https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
# It basically shows usage of calibration data for rectifying the fisheye.
# As and 'FYI', for a more generalized example for intrinsic correction using open CV, see:
# https://github.com/elcorto/unfish/blob/master/unfish/calc.py
import sys
import os
import numpy as np
import json
import glob
import argparse as argp
import cv2
from sklearn.metrics import balanced_accuracy_score
assert cv2.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0'


#map1=np.empty()
#map2=np.empty()
map1=[]
map2=[]

###############################################################################
#                                                                             #
#                                                                             #
###############################################################################
def undistort(img_IN, K, D,img_DIM):
    global map1,map2
    h,w = img_IN.shape[:2]
    print(img_DIM, h, w)
    if h != img_DIM[1] or w != img_DIM[0]:
        return []

    if map1==[] and map2==[]:
        DIM = tuple([img_DIM[0:1]])
        print(DIM)
        print("K=", K)
        print("D=", D)
        # Set balance to 0 for minimum unwanted pixels, 1 for max FOV
        # after estimating new camera matrix, balance value does not matter.
        balance=1.0
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=balance)
        print("\nnew_K=", new_K)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (h,w), cv2.CV_16SC2)
        undistorted_img = cv2.remap(img_IN, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=balance)
#        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv.CV_16SC2)

    return undistorted_img

###############################################################################
#                                                                             #
#                                                                             #
###############################################################################
def main(): 

    parser = argp.ArgumentParser(prog=sys.argv[0], description="Rectify fisheye images using estimated intrinsic parameters from the camera calibration.")

    parser.add_argument('-d', '--directory', type=str, required=False, default=[],
                        help='FULL path to directory containing valid JPG or PNG images')
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='Full path to JSON file containing calibration parameters')
    args=parser.parse_args()

#DIM=XXX
#K=np.array(YYY)
#D=np.array(ZZZ)

    calibfile_path=args.file
    fp_JSON = open(calibfile_path,'r')
    calibration_data = json.load(fp_JSON)
    K = np.asarray(calibration_data["K"])
    D = np.asarray(calibration_data["D"])

    images_jpg = glob.glob(args.directory + '/*.jpg')
    images_png = glob.glob(args.directory + '/*.png')
    images_all = images_jpg + images_png
    cv2.namedWindow("fisheye input", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Rectified", cv2.WINDOW_AUTOSIZE)
        
    for img_path in images_all:
        img = cv2.imread(img_path)
        undistorted_img = undistort(img, K, D, img.shape )

        undist_sc = cv2.resize(undistorted_img, (0,0), fx=0.25, fy=0.25 )
        img_sc = cv2.resize(img, (0,0), fx=0.25, fy=0.25 )
        cv2.imshow("fisheye input", img_sc)
        cv2.imshow("Rectified", undist_sc)
        k = cv2.waitKey(0)
        if k==ord('q'):
            break
        elif k==ord('s'):
            undist_img_path = os.path.splitext(img_path)[0] + '_undistorted.png'
            cv2.imwrite(undist_img_path, undistorted_img)
            print("Wrote ", undist_img_path)

    cv2.destroyAllWindows()
    sys.exit(0)

if __name__ == '__main__':
    main()