# Sensor calibration
* © 2026 Regents of the University of Minnesota. All rights reserved.
* This program is shared under the terms and conditions of the GNU Affero General Public License 3.0, License. Further details about the GNU Affero GPL 3.0 license are available in the LICENSE text file.
  
## /code

**process_fisheye_calibdata.py**: Uses JSON corner point detection camera data to estimate
intrinsic and extrinsic camera calibration parameters. Calibration code based on Kannala-Brandt fisheye model openCV implementation [1].

**undistort.py**: Demonstrates image rectification using fisheye camera calibration model.

1. J. Kannala and S. S. Brandt, "A generic camera model and calibration method for conventional, wide-angle, and fish-eye lenses," 
in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 28, no. 8, pp. 1335-1340, Aug. 2006, doi: 
10.1109/TPAMI.2006.153

## /WV-SFv481

This folder contains calibration data from the Panasonic 2992x2992 fisheye camera. 

**File description**:

- **WV-SFV481_2992X2992_fisheye_calib.json**: The intrinsic/extrinsic camera calibration parameter file.

- **WV-SFV481_corner_det_2992X2992.json**: Extracted calibration grid board pixel coordinate data from a multiplicity of captured full resolution 2992x2992 fisheye camera images.

- **SURVEY_CORRESP_ALL-Tue_2017-03-14_123002-171_raw.csv**: Extracted 2D (_x,y_) image to _XYZ_ world coordinate point data from the _distorted_ original full resolution camera image. The image coordinates are in units of pixels. The world coordinates are expressed with respect to the camera pole center location ({_X_,_Y_,_Z_}w), and in Minnesota South State Plane coordinates ({_X_,_Y_,_Z_}spc), in units of US feet. The world point coordinates were derived from Google Earth extracted Latitude, Lognitude coordinates.

- **SURVEY_CORRESP_ALL-Tue_2017-03-14_123002-171_undist.csv**: Extracted 2D (_x,y_) image to _XYZ_ world coordinate point data from the _rectified_ full resolution image. The image coordinates are in units of pixels. The world coordinates are expressed with respect to the camera pole center location ({_X_,_Y_,_Z_}w), and in Minnesota South State Plane coordinates ({_X_,_Y_,_Z_}spc), in units of US feet. 
  

## /WV-SFv481

This folder contains calibration data from the Panasonic 2992x2992 fisheye camera.

**File description**:

- **WV-S4576L_2992x1992.json**: The intrinsic/extrinsic camera calibration parameter file.

- **WV-S4576L_corner_det_2992X2992.json**: Extracted calibration grid board pixel coordinate data from a multiplicity of captured full resolution 2992x2992 fisheye camera images.

- **SURVEY_CORRESP_ALL-vlcsnap-2025-08-20-16h15m57s696_raw.csv**: Extracted 2D (_x,y_) image to _XYZ_ world coordinate point data from the _distorted_ original full resolution camera image. The image coordinates are in units of pixels. The world coordinates are expressed with respect to the camera pole center location ({_X_,_Y_,_Z_}w), and in Minnesota South State Plane coordinates ({_X_,_Y_,_Z_}spc), in units of US feet. The world point coordinates were derived from Google Earth extracted Latitude, Lognitude coordinates.

- **SURVEY_CORRESP_ALL-vlcsnap-2025-08-20-16h15m57s696_undist.csv**: Extracted 2D (_x,y_) image to _XYZ_ world coordinate point data from the _rectified_ full resolution image. The image coordinates are in units of pixels. The world coordinates are expressed with respect to the camera pole center location ({_X_,_Y_,_Z_}w), and in Minnesota South State Plane coordinates ({_X_,_Y_,_Z_}spc), in units of US feet. 
