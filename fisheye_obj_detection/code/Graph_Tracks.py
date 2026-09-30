##########################################################################
#                                                                        #
#  © 2026 Regents of the University of Minnesota. All rights reserved.   #
# This program is shared under the terms and conditions of the GNU       #
# Affero General Public License 3.0, License. Further details about      #
# the GNU Affero GPL 3.0 license are available in the LICENSE text file. #
#                                                                        #
##########################################################################
import glob
import os, sys, math
import json
import datetime
from pathlib import Path
import argparse
import inspect
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numpy.linalg import inv
import cvzone
from typing import List
import struct
from collections import defaultdict

import ctypes

# my stuff:
from TMmodules import TM_trackdata as td
from TMmodules import TM_fisheye as TM_fe
from TMmodules import TM_trajectory as TM_traj
from TMmodules.TM_TrackDataRecords import TM_trackDataRecords
from TMmodules import TM_plottracks as TM_Plots
from TMmodules.TM_plotSpeedHeading import TM_plotSpeedHeading as TM_plotsphd

MAIN_TRACK_COLOR=3
 # alternate colors for each associated track ID within a complete tracklet (primary trackID and its associated track IDs)
assoc_trackID_colors = [ (255, 0, 0), (0, 0, 255), (0, 255, 0), (0,255, 255), (255, 0, 255) ] # red, green, blue, yellow, magenta

plottracks = None
plotspeed_heading = None

###############################################################################
##                                                                           ##
##                                                                           ##  
###############################################################################
def find_xytrack_points(onclicked_data_dict, Xw_all, Yw_all, time_all, frames_all):
    output_dict = {}
    for key, xydata in onclicked_data_dict.items():
        timeval = xydata[0]
        arr = time_all
        idx = np.abs(arr - timeval).argmin()
        closest_time = time_all[idx]
        closest_frame = frames_all[idx]
        closest_X = Xw_all[idx]
        closest_Y = Yw_all[idx]
        output_dict[key] = [ closest_time, closest_frame, closest_X, closest_Y ]
    return output_dict

###############################################################################
##                                                                           ##
##                                                                           ##  
###############################################################################
def plot_xytrackpoints(trackplot_obj, xytracks_and_labels):
    for label, info in xytracks_and_labels.items():
        x=info[2]
        y=info[3]
        trackplot_obj.plottext(x,y,label,fgcolor=(1,1,1),bgcolor=(0,0,1), fontsize=12,)

###############################################################################
##                                                                           ##
##                                                                           ##  
###############################################################################
def clear_plots(tid, assoc_tids:List):
    plottracks.removeTrackedLine(tid)
    plotspeed_heading.removeTrackedLine(tid)
    plotspeed_heading.removeOnClickedLines()
    plottracks.removeText()
    for assoc_tid in assoc_tids:
        plottracks.removeTrackedLine(assoc_tid)
        plotspeed_heading.removeTrackedLine(assoc_tid)


###############################################################################
##                                                                           ##
##  load the world XYZ to image xy correspondence data.                      ##
##                                                                           ##  
###############################################################################
def load_correspondence_pnts( img_XYZ_datafile ):
    if img_XYZ_datafile is not None:
        if img_XYZ_datafile.lower().endswith('.csv'):
            df = pd.read_csv(img_XYZ_datafile)
        elif img_XYZ_datafile.lower().endswith('.xls') or img_XYZ_datafile.lower().endswith('.xlsx'):
            print(f"...opening excel file, {img_XYZ_datafile}...\n")
            df = pd.read_excel(img_XYZ_datafile)
        if df is not None or len(df) > 0:
            pntIDnames = df['pntID']
            return df[['ix', 'iy']].values, df[['Xw','Yw','Zw']].values
        else:
            print(f"{img_XYZ_datafile}:File dataframe input error!")
            return None, None

###############################################################################
##                                                                           ##
##  Graph each track and allow to pick ones to plot all the data.            ##
##                                                                           ##  
###############################################################################
def show_tracks( trackrecords, selected_trackid=-1):
    global plottracks, plotspeed_heading
        # plot the tracks
    RA_outerRadius = 85.0 # ft.
    RA_islandRadius = 30.0 # ft.
    RA_truckskirtRadius = RA_islandRadius + 11.0
    if plottracks is None:
        plottracks = TM_Plots.TM_plottracks( [-180,180], [-180,180], RA_islandRadius,RA_truckskirtRadius,RA_outerRadius)
    if plotspeed_heading is None:
        plotspeed_heading=TM_plotsphd(fps=15)

    start_tracklet_IDs = trackrecords.groupby(['trackID'])['trackID'].first().tolist()
    
    TrackIDs=[]
    
    for cnt, trackid in enumerate(start_tracklet_IDs):
        if selected_trackid > 0 and trackid != selected_trackid:
            continue

        Xw_aggregated=[]
        Yw_aggregated=[]
        time_aggregated=[]
        frames_aggregated=[]

        Xw = list( trackrecords.query("trackID==@trackid and assoc_TrackID==@trackid")['xpos'] )
        Yw = list( trackrecords.query("trackID==@trackid and assoc_TrackID==@trackid")['ypos'] )
        frames = list( trackrecords.query("trackID==@trackid and assoc_TrackID==@trackid")['frame'] )
        Xw_aggregated = Xw
        Yw_aggregated = Yw
        frames_aggregated=frames

        # only interested in longer tracks
        total_track_length = len(list( trackrecords.query("trackID==@trackid")['xpos'] ))
        if total_track_length < 100:
            continue
        plottracks.updatePlot(trackid, Xw,Yw, colordef=(0,0,0))

        #time = np.linspace( frame_rng[0]/fps, frame_rng[1]/fps, len(fulltrack_rec) )
        time = [ f/fps for f in frames  ]
        time_aggregated=time
        heading = list( trackrecords.query("trackID==@trackid and assoc_TrackID==@trackid")['heading'] )
        speed = list( trackrecords.query("trackID==@trackid and assoc_TrackID==@trackid")['speed'] )
        plotspeed_heading.update_timeaxis( np.floor(np.min(time)-1.0), np.ceil(np.max(time)+1) )
        plotspeed_heading.updateHeadingPlot(trackid, time, heading, colordef=(0,0,0))
        plotspeed_heading.updateSpeedPlot(trackid, time, speed,  colordef=(0,0,0))

        tracks_assocID = trackrecords.query("assoc_TrackID != @trackid and trackID == @trackid")['assoc_TrackID'].unique().tolist()
        speed_assoc=[]
        heading_assoc=[]
        Xw_assoc=[]
        Yw_assoc=[]
        frames_assoc=[]
        for cnt,assoc_tid in enumerate(tracks_assocID):
            Xw_assoc=[]
            Yw_assoc=[]
            Xw_assoc = list( trackrecords.query("assoc_TrackID == @assoc_tid and trackID==@trackid")['xpos'])
            Yw_assoc = list( trackrecords.query("assoc_TrackID == @assoc_tid and trackID==@trackid")['ypos'])
            r = 1.0*int(cnt % 2 ==0)
            g = 1.0*int(cnt % 2 != 0)
            plottracks.updatePlot(assoc_tid, Xw_assoc,Yw_assoc, colordef=(r,g,0))
            #print(f"len assoc ID {assoc_tid} line={len(Xw_assoc)}, len track ID {trackid} line={len(Xw)}")

            speed_assoc=[]
            heading_assoc=[]
            frames_assoc  = trackrecords.query("assoc_TrackID == @assoc_tid and trackID==@trackid")['frame'].tolist()
            speed_assoc   = trackrecords.query("assoc_TrackID == @assoc_tid and trackID==@trackid")['speed'].tolist()
            heading_assoc = trackrecords.query("assoc_TrackID == @assoc_tid and trackID==@trackid")['heading'].tolist()
            time_assoc = [ f/fps for f in frames_assoc ]

            Xw_aggregated = [ *Xw_aggregated, *Xw_assoc ]
            Yw_aggregated = [ *Yw_aggregated, *Yw_assoc ]
            frames_aggregated= [ *frames_aggregated, *frames_assoc ]
            time_aggregated = [ *time_aggregated, *time_assoc ]
            
            plotspeed_heading.update_timeaxis( np.floor(np.min([ *time_assoc, *time])-1.0), np.ceil(np.max([ *time_assoc, *time])+1) )
            plotspeed_heading.updateHeadingPlot(assoc_tid, time_assoc, heading_assoc, colordef=(r,g,0))
            plotspeed_heading.updateSpeedPlot(assoc_tid, time_assoc, speed_assoc,  colordef=(r,g,0))
            
        if selected_trackid > 0:
            TrackIDs.append([trackid, tracks_assocID] )

        resp = input(f"track ID={trackid}, assoc IDs={tracks_assocID},enter any key to plot points on RA diagram:")
        onclicked_data_dict = plotspeed_heading.get_ClickedDataPoints()
        xytrack_points_from_onclicked_data = find_xytrack_points(onclicked_data_dict, Xw_aggregated, Yw_aggregated, time_aggregated, frames_aggregated)
        print(xytrack_points_from_onclicked_data)
        plot_xytrackpoints(plottracks, xytrack_points_from_onclicked_data)
        resp = input(f"track ID={trackid}, assoc IDs={tracks_assocID}, enter (k)eep, (q)uit, or <CR>")
        if resp=="q" or resp=='k':
            TrackIDs.append([trackid, tracks_assocID] )
            if resp=="q":
                clear_plots(trackid, tracks_assocID)
                break
        clear_plots(trackid, tracks_assocID)
                
    return TrackIDs


###############################################################################
##                                                                           ##
##                        main                                               ##
##                                                                           ##  
###############################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Vehicle and pedestrian classes trajectory extractor")
    parser.add_argument('--pixel2world_file', type=str, help="xcel file listing: pntID,ix,iy,Xw,Yw,Zw image to world point correspondences")
    parser.add_argument('--calib_file', '-c', type=str, default='./fisheye_calib.json',
                         help="Optional full path name of JSON camera calibration file (default=\"fisheye_calib.json)")
    parser.add_argument("--video_out",'-o', type=str,default="NOT_SPECIFIED",help ="optional PROCESSED mp4 video output name (default=videoout.mp4)")
    parser.add_argument('--video_dir', '-d', type=str, help="Directory containing UNPROCESSED video")
    parser.add_argument('--video_file', '-f', type=str, help="video filename")
    parser.add_argument('--trajectorydata_path', '-p', type=str, default='', help="path to binary or feather file containing trajectory data")
    parser.add_argument('--skip_speed', type=bool, default=False, help="skip processing speed and heading data if not needed")
    parser.add_argument('--bound_trajectories', '-b', action="store_true", default=False, help="option to filter out trajectories with positions that are outside of a given radius from the center of the roundabout")
    parser.add_argument('--trajectory_file', '-t', type=str, 
                        help="file name of the binary formatted data, or pandas .feather format file containing the trajectory data.")
    parser.add_argument('--track_list', '-l', type=str, default='', help="text file list of track ids to examine.")
    args = parser.parse_args()
    
    trackid_list_file = args.track_list
    TrackID_list = []
    if len(trackid_list_file) > 0 and os.path.isfile(trackid_list_file):     
       with open(trackid_list_file, "r", encoding="utf-8") as file:
    #with open(trackid_list_file, "r") as file:
            file.seek(0)
            for line in file:
                TrackID_list.append( int( line.strip() ) )
    elif len(trackid_list_file) > 0:
        print(f'Error finding or opening {trackid_list_file}!')
    print(f'track ID list: {TrackID_list}')
     
    video_data = os.path.join( args.video_dir, args.video_file )
    # Open the video source 

    fps=15.0
    cap = cv2.VideoCapture(video_data)
    if not cap.isOpened():
        print(f"the input video source or file provided, {video_data}, does not exist or is invalid!")
        sys.exit(1)

    #pixel_to_world_filename = args.pixel2world_file
    video_out = args.video_out
    output_video = False
    if video_out != 'NOT_SPECIFIED':
        # Retrieve frame width, height, and FPS
        output_video = True
        frame_width = 640  # int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = 640 # int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 15.0
        # Define codec and create VideoWriter (mp4v for .mp4)
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4v for .mp4 works with opencv out of the box.
        fourcc = cv2.VideoWriter_fourcc(*'avc1') # h.264 compliant codec, better compression and works on most platforms
        vfout = cv2.VideoWriter(video_out, fourcc, fps, (frame_width, frame_height))
        if not vfout.isOpened():
            print(f"\n**** failed to create video writer for {video_out}!!")
            sys.exit(1)

    video_data = os.path.join( args.video_dir, args.video_file )
    calibration_filename = args.calib_file
    pixel_to_world_filename = args.pixel2world_file
    skip_speed = args.skip_speed

    xy_imagepnts, XYZ_worldpnts  = load_correspondence_pnts( pixel_to_world_filename )    
    feye = TM_fe.TM_fisheye()
    feye.load_calibration(calibration_filename)
    # resize undistortion map to a different image resolution
    # tradeoff between processing speed and image quality and reprojection accuracy
    # (1280 is as sharp as the full res when displayed at 640)
    feye.resize(1280)
    Kmtx, Dcoeff = feye.getKD()

    trajectory = TM_traj.TM_trajectory(XYZ_worldpnts, xy_imagepnts,Kmtx, Dcoeff )
    # assumes using the undistorted image coordinates
    trajectory_undist = TM_traj.TM_trajectory(XYZ_worldpnts, xy_imagepnts,Kmtx, 0, sf_IN=float(1280/2992) )
    trajectory_undist.setScaleFactor(float(640/1280)) # useful for adjusting to rescaled image sizes 

    # input track data corresponding to this video
    tracking_data = TM_trackDataRecords(args.trajectorydata_path, args.trajectory_file)
    datarecords = tracking_data.ReadData()
    datarecords['speed'] *= 3600.0/5280.0
    datarecords['heading'] *= 180.0/np.pi
    #tracking_data.SaveData()
    if 'speed' not in tracking_data.col_names and 'heading' not in tracking_data.col_names:
        print("speed and heading data not found in the track data file, skipping processing of speed and heading data...")
        skip_speed = True

    # step through tracks, or select one to visualize.
    if len(TrackID_list) == 0:
        selected_tracks = show_tracks( datarecords, selected_trackid=-1)

    else:
        # plot the selected tracks
        for selected_track in TrackID_list:
            selected_tracks_ret = show_tracks( datarecords, selected_track)

    # go to video portion 

