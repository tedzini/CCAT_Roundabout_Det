# Fisheye Object Detection 
* © 2026 Regents of the University of Minnesota. All rights reserved.
* This program is shared under the terms and conditions of the GNU Affero General Public License 3.0, License. Further details about the GNU Affero GPL 3.0 license are available in the LICENSE text file.

## Code description

See corresponding shell/batch scripts for example usage.

* **Graph_Tracks.py**: Inputs saved feather/binary track files to examine specific tracks or 'page' through each object detection track. The user can then click specific time points within heading/speed graphs, which will then identify and locate the points within the XY trajectory graph.

* **TM_output_detections.py**: Executes real-time detection with specified YOLOv11 Oriented Object Bounding (obb) trained task. For an example of required inputs, and the various options, refer to the corresponding bash shell script. To list a description of all input arguement specifications, run:
 ```sh
python ./TM_output_detections.py -h
  ```
 Note the option to store a bineary formatted track detection file. For a more efficient open source file specification, it can be converted to a feather file.

* **Overlay_All_Tracks.py**: Examine, and alternatively output a labeled video, all detection tracks for a specific video. This requires a recorded video file and the associated *.feather or *.bin track data file. 

* **convert2feather.py**: Standalone code for reading and converting object tracks binary file to a standard feather file.