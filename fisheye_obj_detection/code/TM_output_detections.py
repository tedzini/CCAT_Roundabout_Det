import glob
import os, sys, math
import json
from datetime import datetime
import argparse
import inspect
import cv2
import numpy as np
import pandas as pd
from numpy.linalg import inv
import cvzone
from typing import List
import struct
import ctypes

# present on Jetson platforms
try:
    import vpi
    vpi_found = True
except ModuleNotFoundError:
    vpi_found = False
    print("module 'vpi' not found!")

# DL modules
try:
    from ultralytics import YOLO
    ultralitics_found = True
except ModuleNotFoundError:
    ultralitics_found = False
    print("module 'ultralitics' not found!")

torch_found=False
try:
    import torch
    torch_found = True
except ModuleNotFoundError:
    torch_found = False
    print("module 'torch' not found!")

# custom modules:
from TMmodules import TM_plottracks as TM_Plots
from TMmodules import TM_KFwrapper as KF
from TMmodules import TM_fisheye as TM_fe, TM_trajectory as TM_traj
from TMmodules import TM_trackdata as td
from TMmodules import TM_BayesMeanCov_update as TMbmc

# ////////////////////////////////////////////////////////////////////////////////////
# //                                                                                //
# //        Utility for tallying net timing performance metrics                     //
# //                                                                                //
# ////////////////////////////////////////////////////////////////////////////////////
class TimingPerf:
	def __init__(self):
		self.t_preproc = TMbmc.TM_mean_cov_update()
		self.t_inf     = TMbmc.TM_mean_cov_update()
		self.t_postproc= TMbmc.TM_mean_cov_update()
		return
		
	# input the model results to query
	def update(self, model_result):
		compperf = model_result[0].speed # inference speed, NMS speed, and total time per image
		self.t_preproc.update(compperf['preprocess'])
		self.t_inf.update(compperf['inference'])
		self.t_postproc.update(compperf['postprocess'])
		return

	def Count(self):
		return self.t_preproc.Count()

	def show_timings(self):
		N = int(self.t_preproc.Count())
		t_preproc = self.t_preproc
		t_inference = self.t_inf
		t_postproc = self.t_postproc
		
		mn_preproc = t_preproc.Mean()
		std_proproc = t_preproc.stdev()
		print(f'N={N:5d}, Preprocess  mn={t_preproc.Mean()[0]:.2f} +/-{t_preproc.stdev()[0]:.2f} msec')
		print(f'N={N:5d}, Inference   mn={t_inference.Mean()[0]:.2f} +/-{t_inference.stdev()[0]:.2f} msec')
		print(f'N={N:5d}, Postprocess mn={t_postproc.Mean()[0]:.2f} +/-{t_postproc.stdev()[0]:.2f} msec')
		return

KFassign_cnt = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
tracks_dict = {}
KFassing_dict = {}
KFdim_dict = { 'car' : 6, 'bus' : 6, 'truck' : 6, 'person' : 4, 'bicycle' : 4 }
last_heading = 0


# ////////////////////////////////////////////////////////////////////////////////////
# //                                                                                //
# //  data dictionary for tracklets                                                 //
# //                                                                                //
# ////////////////////////////////////////////////////////////////////////////////////
ID_EL=td.ID_EL
CLASS_EL=td.CLASS_EL
FRAMECNT_EL=td.FRAMECNT_EL
LOSTCNT_EL=td.LOSTCNT_EL
INBOUNDS_EL=td.INBOUNDS_EL
XYPOS_EL=td.XYPOS_EL
SPEED_EL = td.SPEED_EL
HEADING_EL = td.HEADING_EL
KF_EL = td.KF_EL

def in_polygon(result,in_left,in_right,in_top,in_bottom):
    return in_left > 0 or in_right > 0 or in_top > 0 or in_bottom > 0

def output_window_callback(event, x, y, flags, param):
     if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
#        print(point)


# ////////////////////////////////////////////////////////////////////////////////////
# //                                                                                //
# //  Kalman filter predictions for detected object                                 //
# //                                                                                //
# ////////////////////////////////////////////////////////////////////////////////////
def  kf_predict_innovate(kf_hdl, XY_in ):
    kf_hdl.predict()
    kf_hdl.update_measures( XY_in )
    est_xy, _ = kf_hdl.get_estimate()
    if len(est_xy) > 1:
        XYw_est = np.asarray(est_xy[0:2])
        update = True
    else:
        est_xy = np.asarray( XY_in )
        update=False

    return(XYw_est, update)

# ////////////////////////////////////////////////////////////////////////////////////
# //                                                                                //
# //  manage track corrections                                                      //
# //                                                                                //
# ////////////////////////////////////////////////////////////////////////////////////
def  check_trackID(trackID_IN: int, classname_IN, current_XYw, radboundary, frame_cnt):
    global last_frame_cnt
    global tracks_dict
         
    removed_trackID = None
    
    # assume peds min speed is slower than vehicles
    # 25 ft/(30*(1/15)) ~= 12.5 ft/sec = 8.5 mph
    # 10 ft/(30*(1/15)) ~= 3.4 ft/sec
    if classname_IN != 'person' or classname_IN != 'bicycle':
        dead_track_maxdistance = 25.0
    else:
       dead_track_maxdistance = 10.0
        
    XYw = np.asarray(current_XYw)
    XYw.reshape(2,)
    
    #if trackID_IN==67:
    #    print(f'\nAt top of  check_trackID; id={trackID_IN} {XYw[0]}. {XYw[1]}:')
    if XYw[0]*XYw[0] + XYw[1]*XYw[1] <= radboundary*radboundary:
        inbounds = True
    else:
        inbounds = False
        
    # update num consecutive frames lost for current list of tracks
    for tID, data in tracks_dict.items():
        lost_cnt = frame_cnt - data[FRAMECNT_EL]
        if tID != trackID_IN:
            data[LOSTCNT_EL] = lost_cnt
            tracks_dict[tID] = data

    # get rid of 'aged out' tracks 
    for tID in list(tracks_dict):
        data = tracks_dict[tID]
        lost_cnt = data[LOSTCNT_EL]
        if lost_cnt > 30:
            del tracks_dict[tID]
            removed_trackID=tID

    # update track data for existing track
    if trackID_IN in tracks_dict:
        data = tracks_dict[trackID_IN]
        kf_handle = data[KF_EL]
        xy_est, _ = kf_predict_innovate(kf_handle, XYw)
        data[XYPOS_EL] = xy_est
        data[FRAMECNT_EL] = frame_cnt
        data[INBOUNDS_EL] = inbounds
        data[LOSTCNT_EL] = 0
        data[CLASS_EL] = classname_IN # ideally, should not change.
        tracks_dict[trackID_IN] = data
        return tracks_dict, trackID_IN, removed_trackID
    
    # was trackID a re-associated track?
    for tID, data in tracks_dict.items():
        associated_trackID=data[ID_EL]
        if associated_trackID == trackID_IN:
            data[XYPOS_EL] = XYw
            kf_handle = data[KF_EL]
            xy_est, updated = kf_predict_innovate(kf_handle, XYw)
            if trackID_IN==67:
                print(f"Estimated XY {len(XYw_est)}:", XYw_est)
            data[XYPOS_EL] = xy_est

            data[FRAMECNT_EL] = frame_cnt
            data[INBOUNDS_EL] = inbounds
            data[LOSTCNT_EL] = 0
            data[CLASS_EL] = classname_IN # ideally, should not change.
            tracks_dict[tID] = data
            removed_trackID = trackID_IN
            return tracks_dict, tID, removed_trackID
                
    # might new track be a continuation of a lost track that isn't aged out?
    if inbounds:
        for  tID, data in tracks_dict.items():
            lost_cnt = data[LOSTCNT_EL]
            if (lost_cnt < 2):
                continue
            
            # update state pred. for object associated with tID 
            # w/o measurement using KF a-priori state prediction :
            kf_handle = data[KF_EL]
            kf_handle.predict_steadystate()
            est_xy, _ = kf_handle.get_prediction()
            
            within_boundary = data[INBOUNDS_EL]
            if not within_boundary:
                continue

            q_XYw = data[XYPOS_EL]
            XYw_list = XYw.tolist()
            x=XYw[0]
            y=XYw[1]
            dXY = np.array( [ XYw[0] - q_XYw[0], XYw[1] - q_XYw[1] ] )
            dXY.reshape(2,)
            #print(f"XYw={XYw[0]},{XYw[1]}, data[XYPOS_EL]={q_XYw}")
            cname = data[CLASS_EL]
            print(f"checking tID={int(tID)} with input ID={int(trackID_IN)} with lost cnts={int(lost_cnt)}, dX,dY={dXY}, dist={np.linalg.norm(dXY):.2f}, {within_boundary}")
            if np.linalg.norm(dXY) < dead_track_maxdistance and classname_IN == cname:
                print(f"Object type:{cname}, lost track ID#{int(tID)} ({int(lost_cnt)}) is close to new track ID#{int(trackID_IN)}!")
                print(f"dist = {np.linalg.norm(dXY):.2f}")

                data[XYPOS_EL] = XYw
                kf_handle = data[KF_EL]
                xy_est, updated = kf_predict_innovate(kf_handle, XYw)
                print(f"Estimated XY {len(est_xy)}:", est_xy)
                data[XYPOS_EL] = xy_est
                data[FRAMECNT_EL] = frame_cnt
                data[LOSTCNT_EL] = 0
                data[ID_EL]=trackID_IN
                tracks_dict[tID] = data
                removed_trackID = trackID_IN
                return tracks_dict, tID, removed_trackID
    
    # no dead-tracks were close enough
#    print(f'New track {trackID} added!')
    
    # Add New unassociated Tracklet. Notice how the associated TrackID is set to itself
    if trackID_IN==67:
        print(f"OK for {trackID_IN} is being added, with x,y={XYw[0]},{XYw[1]}")
    newdata = [None]*td.TRACK_DATA_REC_SZ
    newdata[ID_EL]=trackID_IN
    newdata[CLASS_EL]=classname_IN
    newdata[FRAMECNT_EL]=frame_cnt
    newdata[LOSTCNT_EL]=0
    newdata[INBOUNDS_EL]=inbounds
    newdata[XYPOS_EL]=XYw
    newdata[SPEED_EL]=-9999.  # updated in app 
    newdata[HEADING_EL]=-9999. # updated in app
    kf_handle = KF.TM_KFwrapper(state_dim=KFdim_dict[classname_IN], dt=(1./fps), detclass=classname_IN, trackID_IN=trackID_IN) 
    kf_handle.init_state(XYw)
    newdata[KF_EL]=kf_handle
    tracks_dict[trackID_IN] = newdata
    return tracks_dict, trackID_IN, removed_trackID
    
# ////////////////////////////////////////////////////////////////////////////////////
# //                                                                                //
# //  ang2heading()  converts cartesian angle 0 -> +/-180 into compass Heading      //
# //                                                                                //
# ////////////////////////////////////////////////////////////////////////////////////
def ang2heading( ang_IN: float ) -> float:
    if ang_IN < 0.:
        ang_IN = ang_IN + 2*math.pi

    PI_half = math.pi*0.5
    # convert pi/2 => 0 deg. North, 0 => +pi/2 east, 180 deg. => -pi/2 west, 270 => (+/-)pi South
    if ang_IN <= PI_half and ang_IN >= 0.:
        heading_OUT = PI_half - ang_IN
    elif (ang_IN > PI_half and ang_IN  < 3*PI_half):
        heading_OUT = -(ang_IN - PI_half)
    else: # ang_IN >= 3*pi/2
        heading_OUT = math.pi - (ang_IN - 3*PI_half);
    return heading_OUT
 
# ////////////////////////////////////////////////////////////////////////////////////
# //                                                                                //
# //  main()                                                                        //
# //                                                                                //
# ////////////////////////////////////////////////////////////////////////////////////
if __name__ == "__main__":

    global fps
    
    parser = argparse.ArgumentParser(description="VRU Detection and Tracking in Roundabout Scenario")
    parser.add_argument('--model', '-m', type=str, default='./obb/best.pt', help="Path and file name to the YOLO model file")
    parser.add_argument('--video_dir', '-d', type=str, help="Directory containing UNPROCESSED video")
    parser.add_argument('--video_file', '-f', type=str, help="video filename")
    parser.add_argument('--pixel2world_file', '-p', type=str, help="Excel file listing: pntID,ix,iy,Xw,Yw,Zw image to world point correspondences")
    parser.add_argument('--calib_file', '-c', type=str, default='./fisheye_calib.json',
                         help="Optional full path name of JSON camera calibration file (default=\"fisheye_calib.json)")
    parser.add_argument("--video_out",'-o', type=str,default="NOT_SPECIFIED",help ="optional PROCESSED mp4 video output name (default=videoout.mp4)")
    parser.add_argument("--save_trajectories",'-s',action="store_true", help="enable saving the trajectory and detection data")
    parser.add_argument("--plot_trajectories",action="store_true", help="enable to draw and update a plot of the trajectories on a graph")
    parser.add_argument("--traj_file", type=str, default='./trajectories_output.csv', help="Trajectories output file name (default name is: trajectories_output.csv)")
    parser.add_argument("--batch_mode", '-b', action="store_true", default=False, help="specify non-interactive full batch processing of the file or stream.")
    parser.add_argument('--evalperf', '-e', action="store_true", default=False, help="Evaluate pre/post processing and inference time performance metrics.")

    args = parser.parse_args()

    dlmodel_filename = args.model
    video_data = os.path.join( args.video_dir, args.video_file )
    calibration_filename = args.calib_file
    pixel_to_world_filename = args.pixel2world_file
    video_out = args.video_out
    if video_out == 'NOT_SPECIFIED':
        video_out = args.video_file
    plot_trajectories = args.plot_trajectories
    save_trajectories = args.save_trajectories
    trajectory_fname = args.traj_file
    interactive_mode = not args.batch_mode # if not specified, default stores false
    evaluate_time_performance = args.evalperf
    output_window_title = 'YOLO Detection and Tracking'

    if save_trajectories:
#        traj_fh = open(args.video_file[:len(args.video_file)-4] + ".csv","w")
        trajectory_fname = args.video_file[:len(args.video_file)-4] + "-SDSM.bin"
        if os.path.exists(trajectory_fname):
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            trajectory_fname = args.video_file[:len(args.video_file)-4] + f"-SDSM_{timestamp}.bin"
        
        print(f"trajectory output file name: {trajectory_fname}")
        traj_fh = open(trajectory_fname,"wb")
        packed_header = struct.pack(td.HEADER_FORMAT, td.header)
        packed_header = struct.pack(td.HEADER_FORMAT_SDSM, td.header_SDSM)
        traj_fh.write(packed_header)
    else:
        traj_fh = None

    # TODO: make these parameters
    RA_outerRadius = 85.0 # ft.
    RA_islandRadius = 30.0 # ft.
    RA_truckskirtRadius = RA_islandRadius + 11.0

    if plot_trajectories and interactive_mode:
        plottracks = TM_Plots.TM_plottracks( [-140,140], [-140,140], RA_islandRadius,RA_truckskirtRadius,RA_outerRadius)
    else:
        plottracks = None
    
# filen = 'SURVEY_CORRESP_ALL-Tue_2017-03-14_123002-171_raw.xlsx'
# pathname ='C:\home\trucker\Projects\CCAT_CAV_Roundabout_BSM\sensorcalib\WV-SFV481\images\site';
    xy_imagepnts=[]
    XYZ_worldpnts=[]
    if pixel_to_world_filename is not None:
        if pixel_to_world_filename.lower().endswith('.csv'):
            df = pd.read_csv(pixel_to_world_filename)
        elif pixel_to_world_filename.lower().endswith('.xls') or pixel_to_world_filename.lower().endswith('.xlsx'):
            print(f"...opening excel file, {pixel_to_world_filename}...\n")
            df = pd.read_excel(pixel_to_world_filename)
        if df is not None or len(df) > 0:
            pntIDnames = df['pntID']
            xy_imagepnts = df[['ix', 'iy']].values
            XYZ_worldpnts = df[['Xw','Yw','Zw']].values   
    
    
    feye = TM_fe.TM_fisheye()
    feye.load_calibration(calibration_filename)

    Kmtx, Dcoeff = feye.getKD()
    trajectory = TM_traj.TM_trajectory(XYZ_worldpnts, xy_imagepnts,Kmtx, Dcoeff )
    # assumes using the undistorted image coordinates
    trajectory_undist = TM_traj.TM_trajectory(XYZ_worldpnts, xy_imagepnts,Kmtx, 0 )

    if torch_found:
        if torch.cuda.is_available():
            GPU_available=True
            torch.cuda.set_device
            print("GPU is available...")
        else:
            GPU_available=False
    
    
    infmodel=None
    if ultralitics_found:
        infmodel = YOLO(dlmodel_filename, task="obb", verbose=False)
        names=infmodel.names
    print(names)

#{0: 'bus', 1: 'car', 2: 'person', 3: 'truck'}

    # Open the video source 
    cap = cv2.VideoCapture(video_data)
    if not cap.isOpened():
        print(f"the input video source or file provided, {video_data}, does not exist or is invalid!")
        sys.exit(1)
    #out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60, (640,640))
         
    if interactive_mode:
        cv2.namedWindow(output_window_title)
        #cv2.namedWindow('xwalk_W')
        #cv2.setMouseCallback(output_window_title, output_window_callback)
    
    frame_cnt=0
    save_img = False
    avi = video_data.lower().find('.avi')
    rtsp = video_data.lower().find('rtsp')
    if avi > 0:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        total_frames=-1
    outputvid_cnt=1
    total_tracks_records = 0 # number of records when storing tracks
    recording_video=False
    pause = False
    vidout = None
    fps = 15.0

    filters_veh =  [ KF.TM_KFwrapper(state_dim=6, dt=(1./fps), detclass='car') for n in range(200) ]
    filters_bus =  [ KF.TM_KFwrapper(state_dim=6, dt=(1./fps), detclass='bus') for n in range(50) ]
    filters_chv =  [ KF.TM_KFwrapper(state_dim=6, dt=(1./fps), detclass='truck') for n in range(50) ]
    filters_ped =  [ KF.TM_KFwrapper(state_dim=4, dt=(1./fps), detclass='person') for n in range(50) ]
    filters_bike = [ KF.TM_KFwrapper(state_dim=4, dt=(1./fps), detclass='bicycle') for n in range(50) ]
    filter_dict = { 'car': filters_veh, 'bus': filters_bus, 'truck': filters_chv, 'person' : filters_ped, 'bicycle' : filters_bike }
    KFassign_dict = { 'car' : 0, 'bus' : 1, 'truck' : 2, 'person' : 3, 'bicycle' : 4 }
    Zw_offset_dict = { 'car' : -2.5, 'bus' : -5.5, 'truck' : -5.5, 'person' : -2.75, 'bicycle' : -2.75 }

    while cap.isOpened():
        
        if frame_cnt==5 and interactive_mode:
            while cv2.waitKey(5) != ord(' '):
                continue

        ret, frame = cap.read()
        if not ret:
            print("WARNING: Can't receive frame (stream end?). Exiting ...")
            break

        frame_cnt += 1
        if frame_cnt == 60*2*fps:
            print(f"video length reached {frame_cnt}. Exiting...")
            break

        frame_undist = feye.undistort_image(frame)
        #xo,yo, w,h = 875,460,640,480
        #xwalk_W_img = frame_undist[yo:yo+h,xo:xo+w]   # ROI of a cross-walk area
        #cv2.imshow('xwalk_W', xwalk_W_img)

        # test points
        # ix1,iy1 = trajectory.world2pixel(75.0,75.0,0.0)
        # ix2,iy2 = trajectory.world2pixel(75.0,75.0,10.0)

        #frame_undist_dwg = cv2.line(frame_undist_dwg, pt1, pt2, (0,0,255),thickness)

        frame_undist2 = cv2.resize(frame_undist, (640,640))

        scalefac = 640.0/frame.shape[0]
        trajectory_undist.setScaleFactor(scalefac)

        """
        # The data below was used to test the PnP solution and verify directionality of Z-depth.
        ix1_u,iy1_u = trajectory_undist.world2pixel(75.0,75.0,0.0,returnint=True)
        ix2_u,iy2_u = trajectory_undist.world2pixel(75.0,75.0,10.0,returnint=True) # Z goes toward camera
        ix3_u,iy3_u = trajectory_undist.world2pixel(75.0,75.0,-10.0,returnint=True) # Z goes away from camera
        pt1 = (ix1_u,iy1_u)
        pt2 = (ix2_u,iy2_u)
        pt3 = (ix3_u,iy3_u)
        thickness = 3
        #print(f"int UNDISTORTED: {pt1} to {pt2}")
        cv2.line(frame_undist2, pt1, pt2, (0,0,255),thickness)
        cv2.line(frame_undist2, pt1, pt3, (0,255,0),thickness)
        cv2.circle(frame_undist2, pt1,4,(255,0,0),-1)
        
        # check homography
        #(Xw,Yw) = trajectory_undist.pixel2worldXY_homog(pt1[0],pt1[1])
        #print(f"pt1 actual X,Y,Z= 75.0,75.0,0.0, reconstructed X,Y,Z= {Xw},{Yw},0")
        """
        
        if infmodel is not None:
            #inference = model.track(frame,conf=0.5,iou=0.35)
            inference = infmodel.track(frame_undist2,conf=0.35,persist=True, iou=0.35,
                                       tracker="bytetrack.yaml")

            # use this for evaluating inference performance
            if evaluate_time_performance:
                timing_perf.update(inference)
                if timing_perf.Count() == 1000:
                    timing_perf.show_timings()
				
            boxes = inference[0].obb.xyxy.int().cpu().tolist()
            obbs = inference[0].obb.xyxyxyxy.int().cpu().tolist()
            class_ids = inference[0].obb.cls.int().cpu().tolist()
            confidences = inference[0].obb.conf.cpu().tolist()
            if inference[0].obb.id is not None:
                track_ids = inference[0].obb.id.cpu().tolist()

            for box, class_id, objID, confscore in zip(boxes, class_ids, track_ids, confidences):
                classname = names[class_id]
                #x1, y1, x2, y2 = box
                boxLL = box[0:2]
                boxUR = box[2:4]
                cx_box = math.ceil( (boxLL[0] + boxUR[0])*.5)
                cy_box = math.ceil( (boxLL[1] + boxUR[1])*.5)
                # if objID >=4 and objID <= 40:
                #     cv2.rectangle(frame_undist2, boxLL, boxUR, color=(255, 255, 0), thickness=2)
                #     cv2.putText(frame_undist2, "{:d}".format(int(objID)), tuple(boxUR), 
                #                 fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.0, color=(255,255,0))
                # else:
                cv2.rectangle(frame_undist2, boxLL, boxUR, color=(0, 0, 255), thickness=1)
                #cv2.circle(frame_undist2, (cx_box,cy_box), 4, color=(0, 255, 255),thickness=-1)
                Xwp,Ywp = trajectory_undist.pixel2worldXY_homog(cx_box, cy_box)
                cx_new,cy_new = trajectory_undist.world2pixel(Xwp[0],Ywp[0],Zw_offset_dict[classname], returnint=True)
                Xwp_new,Ywp_new = trajectory_undist.pixel2worldXY_homog(cx_new,cy_new)
                cv2.circle(frame_undist2, (cx_new,cy_new), 3, color=(0, 0,255),thickness=-1)

                # check/correct track ID prediction from Tracker
                updated_trackdict, correctedID, removedID = check_trackID(objID, classname, [Xwp_new,Ywp_new], RA_outerRadius, frame_cnt)
                

            # Update KF state predictions or estimates for all tracked Objects ('lost' or not)
            for trackid, track_data in updated_trackdict.items():
                xypos  = track_data[XYPOS_EL].reshape(2,)
                Xwp_new = xypos[0]
                Ywp_new = xypos[1]
                classname = track_data[CLASS_EL]

                # update trajectory data with filter estimates
                kf_handle = track_data[KF_EL]
                state_estimate, _ = kf_handle.get_estimate()
                if state_estimate[0]==0 and state_estimate[1]==0 and state_estimate[2]==0 and state_estimate[3]==0:
                    print(f"for ID={trackid}:, lost cnt={track_data[LOSTCNT_EL]}, x,y={Xwp_new:.2f},{Ywp_new:.2f}, X_est={state_estimate}")

                if len(state_estimate) < 4:
                    continue
                est_speed = np.linalg.norm( np.array([state_estimate[2],state_estimate[3]]))

                try:
                    last_heading = est_heading
                except NameError:
                    pass

                try:
                    est_heading = ang2heading( math.atan2( state_estimate[3], state_estimate[2]) )
                except (TypeError, ValueError) as e:
                    est_heading = last_heading

                #track_data[XYPOS_EL]   = np.array([ state_estimate[0], state_estimate[1] ])
                track_data[SPEED_EL]   = est_speed
                track_data[HEADING_EL] = est_heading

                if track_data[LOSTCNT_EL] > 0:
                    continue

                if frame_cnt > 50 and plot_trajectories and plottracks is not None:
                    plottracks.updatePlot(correctedID, state_estimate[0], state_estimate[1], ignorelength=15)
                    if removedID is not None:
                        plottracks.removeTrackedLine(removedID)

                cx_est,cy_est= trajectory_undist.world2pixel(state_estimate[0],state_estimate[1],0, returnint=True)
                dx_est, dy_est = trajectory_undist.world2pixel(state_estimate[0]+0.2*state_estimate[2],state_estimate[1]+0.2*state_estimate[3],0, returnint=True)
                cv2.circle(frame_undist2, (cx_est,cy_est), 3, color=(0,0,255),thickness=-1)
                cv2.arrowedLine(frame_undist2, (cx_est,cy_est), (dx_est,dy_est), color=(255,255,255),thickness=1,tipLength=0.25)


            # update predictions for 'lost' tracks
            last_frame_cnt = frame_cnt
            # for obb_rect, class_id, objID in zip(obbs, class_ids, track_ids):
            #     c = names[class_id]
            #     # polygon is cw starting from x4,y4 
            #     obbpt4 = obb_rect[0]
            #     obbpt1 = obb_rect[1]
            #     obbpt2 = obb_rect[2]
            #     obbpt3 = obb_rect[3]
            
                #print(f"id={objID}, class({class_id})={c}, tensor obb={obbpt1}")
                # cx = obbpt1[0] + obbpt2[0] + obbpt3[0] + obbpt4[0]
                # cy = obbpt1[1] + obbpt2[1] + obbpt3[1] + obbpt4[1]
                # cx = math.ceil(cx*0.25)
                # cy = math.ceil(cy*0.25)

                # polyl = np.array([[obbpt4], [obbpt1], [obbpt2], [obbpt3]], np.int32)
                # polyl = polyl.reshape(-1,1,2)
                # cv2.polylines(frame_undist2, [polyl], isClosed=True, color=(0, 255, 0), thickness=1)

            # center values are identical
            #print(f"box center={cx_box},{cy_box}\tobb center={cx},{cy}")
 
        # initial connection to live stream or file has a few seconds where the decoding is not stable, 
        # so wait for N frames before saving trajectories and plotting
        if save_trajectories and frame_cnt > 3*fps:
            total_tracks_records = total_tracks_records + len(updated_trackdict)
            for id, track_data in updated_trackdict.items():
                xypos  = track_data[XYPOS_EL].reshape(2,)
                if track_data[INBOUNDS_EL]:
                    inbounds = 'T'
                else:
                    inbounds = 'F' # <ii7sIH1sff

                try:
                    #packed_data = struct.pack(td.DATA_FORMAT, int(id), int(track_data[ID_EL]),track_data[CLASS_EL].encode('utf-8'),
                    #                        int(track_data[FRAMECNT_EL]), track_data[LOSTCNT_EL], inbounds.encode('utf-8'), xypos[0], xypos[1])
                    packed_data = struct.pack(td.DATA_FORMAT_SDSM, int(id), int(track_data[ID_EL]),track_data[CLASS_EL].encode('utf-8'),
                                            int(track_data[FRAMECNT_EL]), track_data[LOSTCNT_EL], inbounds.encode('utf-8'), xypos[0], xypos[1],
                                            track_data[SPEED_EL],track_data[HEADING_EL])
                    traj_fh.write(packed_data)
                except struct.error as e:
                    print(f"\n(frame={frame_cnt}) Error: {e}")
            print(f'num records={total_tracks_records} frame #: {frame_cnt} updated track dict size={len(updated_trackdict)}\n')            

        if recording_video:
            vidout.write(frame_undist2)

        # ---------------------------------------------------------------------
        #--                                                                  --
        #--      User interaction options and display                        --
        #--      Most won't work for rtsp or live stream input since they    --
        #--      require pausing and stepping through frames, but can        -- 
        #--      be used for stored video files.                             --
        #--                                                                  --
        # ---------------------------------------------------------------------
        if not interactive_mode:
            continue

        cv2.imshow(output_window_title, frame_undist2)

        key = cv2.waitKey(5)
        if key & 0xFF == ord('q') or key == 27:
            break

        elif key & 0xFF == ord(' '):
            while cv2.waitKey(5) != ord(' '):
                continue
            continue

        elif key & 0xFF == ord('s'):
            save_img=True
            print("Save image mode enabled...")

        elif key & 0xFF == ord('d'):
            if save_img == False:
                print("NOTE: you must first hit 's' key to put into save mode!\n")
                continue
            if avi > 0:
                filen = video_data[:avi] + "-" + str(frame_cnt) + "_raw.png"
            elif rtsp > 0:
                now = datetime.datetime.now()
                filen = now.strftime("%Y_%m_%d_%H%M%S") + "_raw.png"
            print(f"Saving raw image...file={filen}")
            cv2.imwrite(filen, frame)
            print("DONE")
            save_img=False

        elif key & 0xFF == ord('r'):
            if vidout is not None:
                if recording_video:
                    vidout.release()
                    outputvid_cnt += 1
                    recording_video=False
                    print("...Recording stopped...")
                else:
                    recording_video=True
                    vidout_fname = video_out[:len(video_out)-4] + '-' + str(outputvid_cnt) + ".mp4"
                    vidout = cv2.VideoWriter(vidout_fname, cv2.VideoWriter_fourcc(*'mp4v'), 60, (640, 640))
                    print("...Recording started...")

        elif key & 0xFF == ord('u'):
            if save_img == False:
                print("NOTE: you must first hit 's' key to put into save mode!\n")
                continue
            if avi > 0:
                filen = video_data[:avi] + "-" + str(frame_cnt) + "_undist.png"
            elif rtsp > 0:
                now = datetime.datetime.now()
                filen = now.strftime("%Y_%m_%d_%H%M%S") + "_undist.png"
            print(f"Saving UNDISTORTED image...file={filen}")
            cv2.imwrite(filen, frame_undist)
            print("DONE")
            save_img=False

            # go to a specific point in a stored video stream (e.g. file)
        elif key & 0xFF == ord('j') and total_frames > 0:
            frame_number_in = input("Enter frame number to jump:")
            frame_number = int(frame_number_in)
            if frame_number > 0 and frame_number <= total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        elif key & 0xFF == ord('t') and total_frames > 0:
            timeval = str(input("Enter time (\"HH:MM:SS[.xxx]\" or \"MM:SS[.xxx]\") to jump:")) 
            hrval=0
            hrminsec_pos = [i for i, colon in enumerate(timeval) if colon==':']
            if len(hrminsec_pos)==2:
                pos = hrminsec_pos[0]
                hrval = int(timeval[:pos])
                timeval = timeval[pos+1:]
                pos = hrminsec_pos[1]
            else:
                pos = hrminsec_pos[0]
                print(f'first : position {pos}')
                minuteval = int(timeval[:pos])
                secondval = float(timeval[pos+1:])
                timejump = 3600*hrval + 60*minuteval + secondval
                frame_number = int(timejump*fps)
                if frame_number <= total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                         
    
# # Release the video capture object and close the display window
    #out.release()
    cap.release()
    if save_trajectories:
        traj_fh.close()
    
    if interactive_mode:
        cv2.destroyAllWindows()
