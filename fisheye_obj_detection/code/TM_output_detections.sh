#!/bin/bash -e
##########################################################################
#                                                                        #
#  © 2026 Regents of the University of Minnesota. All rights reserved.   #
# This program is shared under the terms and conditions of the GNU       #
# Affero General Public License 3.0, License. Further details about      #
# the GNU Affero GPL 3.0 license are available in the LICENSE text file. #
#                                                                        #
##########################################################################

#filen = 'SURVEY_CORRESP_ALL-Tue_2017-03-14_123002-171_raw.xlsx'
#pathname ='C:\home\trucker\Projects\CCAT_CAV_Roundabout_BSM\sensorcalib\WV-SFV481\images\site';
PYTHONCMD=$(echo "`which python`")
if [[ ${#PYTHONCMD} -eq 0 ]]; then
    PYTHONCMD=`which python3`
fi

filen="SURVEY_CORRESP_ALL-Tue_2017-03-14_123002-171_raw.xlsx"
filen="SURVEY_CORRESP_ALL-Tue_2017-03-14_123002-317_undist.xlsx"
pathname="C:\\home\\trucker\\Projects\\CCAT_CAV_Roundabout_BSM\\sensorcalib\\WV-SFV481\\images\\site";

videopath=/disk1/videos/portland66_2017/TrafficVideos
videofile=Fri_2017-03-17_153002.avi
videoout=Fri_2017-03-17_153002.mp4

#videofile=Thu_2017-03-16_070003.avi 
#videoout=Thu_2017-03-16_070003.mp4 

export YOLO_VERBOSE=False
#$PYTHONCMD TM_output_detections.py -f "$videofile" -d  "$videopath" --pixel2world_file "$filen" -o "$videoout"
#$PYTHONCMD TM_output_detections.py -f "$videofile" -d  "$videopath" --pixel2world_file "$filen" --plot_trajectories
#$PYTHONCMD TM_output_detections.py -f "$videofile" -d  "$videopath" --pixel2world_file "$filen" -s -b
$PYTHONCMD TM_output_detections.py -f "$videofile" -d  "$videopath" --pixel2world_file "$filen"
echo "...done"
