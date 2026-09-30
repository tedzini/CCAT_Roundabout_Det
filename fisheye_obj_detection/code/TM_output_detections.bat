##########################################################################
#                                                                        #
#  © 2026 Regents of the University of Minnesota. All rights reserved.   #
# This program is shared under the terms and conditions of the GNU       #
# Affero General Public License 3.0, License. Further details about      #
# the GNU Affero GPL 3.0 license are available in the LICENSE text file. #
#                                                                        #
##########################################################################

rem filen = 'SURVEY_CORRESP_ALL-Tue_2017-03-14_123002-171_raw.xlsx'
rem pathname ='C:\home\trucker\Projects\CCAT_CAV_Roundabout_BSM\sensorcalib\WV-SFV481\images\site';
set filen="SURVEY_CORRESP_ALL-Tue_2017-03-14_123002-171_raw.xlsx"
set filen="SURVEY_CORRESP_ALL-Tue_2017-03-14_123002-317_undist.xlsx"
set pathname="C:\\path\\to\\WV-SFV481\\calibration\\data";

python TM_output_detections.py -f "Tue_2017-03-14_123002.avi" -d "C:\\path\\to\\CCAT_CAV_Roundabout_BSM\\videos" --pixel2world_file %filen%
