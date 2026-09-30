# Fisheye Multi-image Object Detection 
* © 2026 Regents of the University of Minnesota. All rights reserved.
* This program is shared under the terms and conditions of the GNU Affero General Public License 3.0, License. Further details about the GNU Affero GPL 3.0 license are available in the LICENSE text file.

## Code description

See corresponding shell/batch scripts for example usage.

* **main_segment.py**: Method for creating the multi-image calibrated rectification output video for the multi-image trained YOLO8 models. This code has not been specifically optimized to execute in real-time on the Jetson Orin platforms.

* **detect.py**: Executes real-time detection with specified YOLOv11 Oriented Object Bounding (obb) trained task. For an example of required inputs, and the various options, refer to the corresponding bash shell script. To list a description of all input arguement specifications, run:
 ```sh
python ./detect.py -h
  ```
