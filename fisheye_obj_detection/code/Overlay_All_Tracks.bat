##########################################################################
#                                                                        #
#  © 2026 Regents of the University of Minnesota. All rights reserved.   #
# This program is shared under the terms and conditions of the GNU       #
# Affero General Public License 3.0, License. Further details about      #
# the GNU Affero GPL 3.0 license are available in the LICENSE text file. #
#                                                                        #
##########################################################################

@ECHO OFF
SET video_path="C:\\videos\\portland66_2017\\TrafficVideos"
SET video_file="Fri_2017-03-17_153002.avi"
SET corresp_file="SURVEY_CORRESP_ALL-Tue_2017-03-14_123002-171_raw.xlsx"
SET corresp_file="SURVEY_CORRESP_ALL-Tue_2017-03-14_123002-317_undist.xlsx"

REM use this if no Feather file exists:q
REM python Tracks_Analysis.py -d 'video_path' -f 'video_file' -t Fri_2017-03-17_153002.bin

REM Tracks_Analysis will store an equivalent Feather formatted file which reads and converts to a table
REM a ton faster than the raw *.bin file
REM bound_trajectories (-b) option filters out trajectories with positions that are outside of a given radius from the center of the roundabout
REM (for analysing 'incomplete tracklets' we only observe trackets when they are within 160' from center of the roundabout, which is about 45 ft. from the crosswalks.)

python Overlay_All_Tracks.py -d %video_path% -f %video_file%  --pixel2world_file %corresp_file% --bound_trajectories -t Fri_2017-03-17_153002-SDSM_2026-06-22_18-33-45.bin

