import sys
import os
import numpy as np
import json
import argparse as argp
import cv2
assert cv2.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0'

# camera specs
# https://i-pro.com/products_and_solutions/en/surveillance/products/wv-sfv481

maxW = 2992 # max res @ 15fps
maxH = 2992
sensorW = 5.54 # active width of imaging sensor
sensorH = sensorW
focallength = 1.38 # mm

###############################################################################
#                                                                             #
# encode numpy arrays in order to store the data the json file                #
#                                                                             #
###############################################################################
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

###############################################################################
#                                                                             #
#                                                                             #
###############################################################################
def second_largest(list_IN):
    if not isinstance(list_IN, list):
        list_p=list(list_IN)
    else:
        list_p=list_IN
    list_p.sort()
    return list_p[-2]

###############################################################################
#                                                                             #
#                                                                             #
###############################################################################
def main():

    parser = argp.ArgumentParser(prog=sys.argv[0], description="Extract and store calibration points")

    parser.add_argument('-d', '--directory', type=str, required=False, default=[],
                        help='FULL path to directory containing the valid JPG calibration images')
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='Full path to JSON file containing 2D pixel image<->3d object pnt correspondences')
    parser.add_argument('-o', '--outputcalib', type=str, required=False, default='WV-SFV481_calib.json',
                        help='Full file path name for storing calibration results')
    args=parser.parse_args()

#obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
    file_path=args.file
    fp_JSON = open(file_path,'r')
    obj_data = json.load(fp_JSON)
    img_list = obj_data["images"]
    Nimages = len(img_list)
    print("processing data from {0} images...".format(Nimages) )

    # All 'valid calibration' images contain the complete grid pattern, and therefore
    # contain the same number of extracted image coordiantes.
    rows = obj_data["imgrows"]
    cols = obj_data["imgcols"]
    imgpoints = np.asarray(obj_data["imgpoints"])
    objpoints = np.asarray(obj_data["objpoints"])
    #print('objpoints:', objpoints.ndim, objpoints.shape, max(objpoints.shape))
    #np.reshape(imgpoints, (105,64,-1))
    #imgpoints.T.reshape(105,64,2)
    #print('\nObject data:\n',objpoints[0:2,0,0:5,:])
    #print('\nImage data:\n',imgpoints[0:2,0:5,0,:])

    # sanity checks...
    NI_samples = max(objpoints.shape)
    NO_samples = max(imgpoints.shape)
    NI_points = second_largest(imgpoints.shape)
    NO_points = second_largest(objpoints.shape)
    assert NI_samples == Nimages, 'Missing extracted image point data; only {0} stored but there are {1} images!'.format(NI_samples,Nimages)
    assert NI_points == NO_points, 'The number correspondences don''t match within each image sample!'
    assert NI_samples == NO_samples, 'The samples associated with number of valid images are not the same between the stored object points and image points'

    print('Number of point correspondances per {0}x{1} image={2}.'.format(cols, rows, NI_points))
    K = np.zeros((3, 3))
    K[0,0] = cols * focallength / sensorW
    K[1,1] = K[0,0]
    K[0,2] = sensorW * 0.5
    K[1,2] = sensorH * 0.5

    #K = np.identity(3)
    D = np.zeros((4, 1))
    print(len(objpoints))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(Nimages)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(Nimages)]

    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    calibration_flags += cv2.fisheye.CALIB_FIX_SKEW
    #calibration_flags += cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
#   #calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
#        gray.shape[::-1],
    rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        (rows, cols),
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-6)
    )
    print("rvecs=",rvecs)
    print("tvecs=",tvecs)
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    rows = obj_data["imgrows"]
    cols = obj_data["imgcols"]

    # save the calibration results
    fd_out = open( args.outputcalib, "w")
    calibresult = {
        "sensorW": 5.54,
        "sensorH": 5.54,
        "focal": 1.38,
        "imgrows": int(rows),
        "imgcols": int(cols),
        "K": K,
        "D": D,
        "RPY": rvecs,
        "t": tvecs
    }
    json.dump(calibresult, fd_out, indent=4, cls=NumpyEncoder)
    

    sys.exit(0)

if __name__ == '__main__':
    main()
