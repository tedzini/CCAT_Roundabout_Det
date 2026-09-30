##########################################################################
#                                                                        #
#  © 2026 Regents of the University of Minnesota. All rights reserved.   #
# This program is shared under the terms and conditions of the GNU       #
# Affero General Public License 3.0, License. Further details about      #
# the GNU Affero GPL 3.0 license are available in the LICENSE text file. #
#                                                                        #
##########################################################################

# obj. tracklet position data
DATA_FORMAT='<ii7sIH1sff'
header='trackID,assoc_TrackID,class,frame,lostcnt,inbounds,xpos,ypos'
HEADER_FORMAT=str(len(header)) + 's'
header=header.encode('utf-8')

# adds obj. speed and heading est. for to satisfy BSM/SDSM elements
DATA_FORMAT_SDSM = DATA_FORMAT + 'ff' 

header_SDSM = 'trackID,assoc_TrackID,class,frame,lostcnt,inbounds,xpos,ypos,speed,heading'
HEADER_FORMAT_SDSM = str(len(header_SDSM)) + 's'
header_SDSM = header_SDSM.encode('utf-8')

# column LUT
recdict = {'trackID': 0, 'assoc_TrackID': 1, 'class': 2, 'frame': 3, 'lostcnt': 4, 'inbounds': 5, 'xpos': 6, 'ypos': 7, 'speed':8, 'heading':9}

# track object ID that has been associated with a previous 'lost track' object ID
ID_EL=0

# string type class name
CLASS_EL=1

# video frame sequence number
FRAMECNT_EL=2

# counts number of frames an existing track object ID is missing
LOSTCNT_EL=3

# a boolean, if within an app specified radial boundary = True, false otherwise
INBOUNDS_EL=4

# current x,y position of the tracked object (or an associated track object ID)
XYPOS_EL=5

# speed
SPEED_EL=6

# heading
HEADING_EL=7

# KF object handle for the tracklet (not considered part of stored record, but included here for reference to the tracklet data structure)
KF_EL=8

#TRACK_DATA_REC_SZ=max(ID_EL,CLASS_EL,LOSTCNT_EL,INBOUNDS_EL,XYPOS_EL)+1
# included speed and heading into tracklet dictionary data
TRACK_DATA_REC_SZ=max(ID_EL,CLASS_EL,LOSTCNT_EL,INBOUNDS_EL,SPEED_EL,HEADING_EL,KF_EL)+1

# parameters used for setting boundaries to monitor lost/missing tracks and plotting
RA_outerRadius = 85.0 # ft.
RA_islandRadius = 30.0 # ft.
RA_truckskirtRadius = RA_islandRadius + 11.0
