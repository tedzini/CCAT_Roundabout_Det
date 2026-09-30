#!/bin/bash
##########################################################################
#                                                                        #
#  © 2026 Regents of the University of Minnesota. All rights reserved.   #
# This program is shared under the terms and conditions of the GNU       #
# Affero General Public License 3.0, License. Further details about      #
# the GNU Affero GPL 3.0 license are available in the LICENSE text file. #
#                                                                        #
##########################################################################

model_path_base=/path/to/your/model/directory
MODEL=${model_path_base}/CCAT_BenS_YOLOv8s.engine
VIDEO_PATH_ROOT=/path/to/your/multiview_video
VIDEO_NAME=Thu_2017-03-16_070003_out_View_W.avi

# change this to False to disable verbose output of detection performance from the YOLOv8 code
export YOLO_VERBOSE=True 
#python detect.py  -m $MODEL -d $VIDEO_PATH_ROOT -f $VIDEO_NAME 2>&1 | tee detect_output.log
python detect.py  -m $MODEL -d $VIDEO_PATH_ROOT -f $VIDEO_NAME
