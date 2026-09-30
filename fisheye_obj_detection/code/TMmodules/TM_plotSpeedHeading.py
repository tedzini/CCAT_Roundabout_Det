##########################################################################
#                                                                        #
#  © 2026 Regents of the University of Minnesota. All rights reserved.   #
# This program is shared under the terms and conditions of the GNU       #
# Affero General Public License 3.0, License. Further details about      #
# the GNU Affero GPL 3.0 license are available in the LICENSE text file. #
#                                                                        #
##########################################################################

import math
from cv2 import data
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('TkAgg') # 'TkAgg' or 'Qt5Agg'
import matplotlib.pyplot as plt
from   matplotlib.patches import Polygon
# ////////////////////////////////////////////////////////////////////////////////////
# //                                                                                //
# //  plot object speed over time for given track                                   //
# //                                                                                //
# ////////////////////////////////////////////////////////////////////////////////////
class TM_plotSpeedHeading:
    fig=None
    ax=None
    lines=[ [-1]*50, [-1]*50 ] # 0 for speed, 1 for heading
    trackid2line = {}
    line_removed=[ [True]*50, [True]*50 ] # 0 for speed, 1 for heading
    line_cnt=0
    cnt=-150
    save_color_IN = None
    clicked_xcoord = None
    clicked_ycoord = None
    mouse_clicked_in_plot = False
    onclick_lineh = []
    onclick_texth = []
    onclick_linecnt=0

    #stores a representation of clicked points along with the label that was
    # displayed for each clicked point.  
    onclicked_data_dict = {}

    # ////////////////////////////////////////////////////////////////////////////////////
    # //                                                                                //
    # //  plot object speed over time for given track                                   //
    # //                                                                                //
    # ////////////////////////////////////////////////////////////////////////////////////
    def __init__(self, min_Speed=0, max_Speed=50, fps=1):

        self.onclick_lineh.append([])
        self.onclick_lineh.append([])
        self.onclick_texth.append([])
        self.onclick_texth.append([])

        # sanity check
        assert max_Speed > min_Speed, "minimum specified speed (MPH) must be greater than maximum Speed (MPH)!"
        plt.ion()
               # use fig, axs = plt.subplots(rows, cols, figsize = (5,5)) for multiple subplots
        self.fig, self.ax = plt.subplots(2,1, sharex=True)
        plt.pause(0.1) 
        plt.show()
        self.fig.set_size_inches(8, 8)
        #self.fig.tight_layout(pad=10.0)
        self.fig.subplots_adjust(hspace=0.35)
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)


        ax = self.ax[0]
        ax.tick_params(labelbottom=True)  # even though the x-axis is shared, we can label the axis values
        ax.set_aspect('auto', adjustable='box')
        self.min_X = 0
        self.max_X = 60 # 60 seconds
        self.min_Y = min_Speed
        self.max_Y = max_Speed
        ax.set_xlim(self.min_X,self.max_X)
        ax.set_ylim(self.min_Y,self.max_Y)
        ax.set_title("Road User Speed")
        ax.set(xlabel="time (sec.)")
        ax.set(ylabel="Speed (MPH)")
        ax.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', axis='both')


        # heading plot
        ax = self.ax[1]
        ax.set_aspect('auto', adjustable='box')
        ax.set_xlim(self.min_X,self.max_X)
        ax.set_ylim(-185, 185)  # Adjust as needed
        ax.set_title(r'Heading (True North =0$^\circ, {\plus}CW \rightarrow East,  {\minus}CCW \rightarrow West$)')
        ax.set(xlabel="time (sec.)")
        ax.set(ylabel="Heading (degrees)")
        ax.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', axis='both')
        
        print('at bottom of TM_plotSpeed init')

    # ////////////////////////////////////////////////////////////////////////////////////
    # //                                                                                //
    # //  returns a representation of clicked points along with the label that was      //
    # //  displayed for each.                                                           //
    # //                                                                                //
    # ////////////////////////////////////////////////////////////////////////////////////
    def get_ClickedDataPoints(self):
        return self.onclicked_data_dict
    
    # ////////////////////////////////////////////////////////////////////////////////////
    # //                                                                                //
    # //  clicking on one of the plots to return the data coordinates.                  //
    # //                                                                                //
    # ////////////////////////////////////////////////////////////////////////////////////
    def onclick(self, event):
        # Ensure click occurs inside a plot area
        if event.inaxes is None:
            return
        
        self.clicked_xcoord = event.xdata
        self.clicked_ycoord = event.ydata

        # indicate x location

        x=[self.clicked_xcoord, self.clicked_xcoord]
        text_str = str(chr(self.onclick_linecnt+65))
        
        for idx, plotax in enumerate(self.ax):
            lineh_el, = plotax.plot( [], [] )
            #lines, = plotax.plot( [], [] )
            #lineh = lines
            y_min, y_max = plotax.get_ylim()
            y=[y_min, y_max]
            lineh_el.set_linewidth(1)
            lineh_el.set_linestyle('--')
            lineh_el.set_color((0,0,0))
            lineh_el.set_xdata(x)
            lineh_el.set_ydata(y)
            texth_el = plotax.text( x=x[0], y=0.95*y_max, s=text_str, 
                            color='white', 
                            weight='bold', 
                            fontsize=12,
                            bbox=dict(facecolor='blue', alpha=0.9, pad=4))
            
                # render the updated plot
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            self.onclick_lineh[idx].append(lineh_el)
            self.onclick_texth[idx].append(texth_el)

        # we only need times
        self.onclicked_data_dict[text_str] = [ x[0], 0 ]
        self.onclick_linecnt +=1

                

    # ////////////////////////////////////////////////////////////////////////////////////
    # //                                                                                //
    # //  plot object speed over time for given track                                   //
    # //                                                                                //
    # ////////////////////////////////////////////////////////////////////////////////////
    def removeOnClickedLines(self):
        self.onclicked_data_dict.clear()
        self.onclicked_data_dict = {}
        
        self.onclick_linecnt = 0

        for i_axis in range(len(self.ax)):
            if self.onclick_lineh[i_axis] == []:
                continue
            line_handles = self.onclick_lineh[i_axis]
            text_handles = self.onclick_texth[i_axis]
            for lineh in line_handles:
                lineh.remove()
            for texth in text_handles:
                texth.remove()
            self.onclick_lineh[i_axis] = []
            self.onclick_texth[i_axis] = []

    # ////////////////////////////////////////////////////////////////////////////////////
    # //                                                                                //
    # //  plot object speed over time for given track                                   //
    # //                                                                                //
    # ////////////////////////////////////////////////////////////////////////////////////
    def setup_linestyle(self, lineh, color_IN=(-1,-1,-1), use_last_color=False):
        lineh.set_linewidth(1)
        lineh.set_linestyle('none')
        lineh.set_linestyle('-')
        lineh.set_marker('o')
        if color_IN is None:
            color_IN = list(np.random.rand(3))
        elif  ( color_IN[0] < 0 or color_IN[1] < 0 or color_IN[2] < 0 ):
            color_IN = list(np.random.rand(3))

        if use_last_color:
            Color_IN = self.save_color_IN
        else: # keep storing the last color
            if self.save_color_IN is None or self.save_color_IN == (-1,-1,-1):
                self.save_color_IN = color_IN
        #print(color_IN)
        lineh.set_color(color_IN)
        lineh.set_markerfacecolor(color_IN)
        lineh.set_markersize(2)

    # ////////////////////////////////////////////////////////////////////////////////////
    # //                                                                                //
    # //  RemoveTrackedLine called by app to remove tracked line associated with        //
    # //  a track that has been lost or is no longer being tracked.                     //
    # //                                                                                //
    # ////////////////////////////////////////////////////////////////////////////////////
    def removeTrackedLine(self,dead_trackID):
        if dead_trackID in self.trackid2line:
            dataset =  self.trackid2line.pop(dead_trackID)
    
            for i_axis in range(len(self.ax)):
                dataref = dataset[i_axis]
                lineindx = dataref[0]
                lineh = self.lines[i_axis][lineindx]
                lineh.remove()
                self.lines[i_axis][lineindx] = -1
                self.line_removed[i_axis][lineindx] = True

    # ////////////////////////////////////////////////////////////////////////////////////
    # //                                                                                //
    # //  update_Xaxis                                                                  //
    # //                                                                                //
    # ////////////////////////////////////////////////////////////////////////////////////
    def update_Xaxis(self, new_min_X, new_max_X):
        self.min_X = new_min_X
        self.max_X = new_max_X
        for ax in self.ax:
            x_min, x_max = ax.get_xlim()
            #print(f"updatePlot: x_min={min(X)}, x_max={max(X)}")
            #if new_min_X <  x_min or max(new_max_X) >= x_max and new_min_X != new_max_X :
            if new_min_X < new_max_X:
                ax.set_xlim( new_min_X, new_max_X)
            ax.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', axis='both')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def update_timeaxis(self, t_min, t_max):
        self.update_Xaxis(t_min, t_max)

    # ////////////////////////////////////////////////////////////////////////////////////
    # //                                                                                //
    # //  updateSpeedPlot                                                               //
    # //                                                                                //
    # ////////////////////////////////////////////////////////////////////////////////////
    def updateSpeedPlot(self,trackID, X, Y, ignorelength=0, colordef=(-1,-1,-1)):
        
        # update the line associated with the track
        axis_index = 0
        ax = self.ax[axis_index]
        ax.grid(True)

        #print('updateSpeedPlot:', f'trackID={trackID}, X={X}, Y={Y}')
        xlist = self.util_ConvertToList(X)
        ylist = self.util_ConvertToList(Y)
        #print(f'updateSpeedPlot: trackID={trackID}, new lengths of X={len(xlist)}, Y={len(ylist)}')
        
        # Sanity check: speed is a +ive scalar, time > 0.
        xlist = [0 if x < 0 else x for x in xlist ]

        # for speeds <= 0, set it to the previous speed value > 0
        for i,speed in enumerate(ylist):
            if speed <= 0:
                ylist[i] = ylist[i-1] if i > 0 else 0
                print(f"Invalid speed value: {speed}. Setting to previous valid value or 0 if no previous valid value exists.")

            # Assuming a sudden change of more than 10 MPH is not plausible in 0.1 sec. (assuming 10 FPS), 
            # we check for sudden large changes in speed values that are not physically plausible
            # In real-time, this check can be done as each new speed value is generated, but for simplicity we do it here before plotting.
            # Note the fastest accelerating car in the world can go from 0 to 60 MPH in 1.4 seconds, which is an acceleration of about 61 MPH per second^2, or 6 MPH in 0.1 second. 
            # So a threshold of 10 MPH per 0.1 second is a reasonable choice to catch outliers while allowing for very aggressive acceleration.
            if i > 0 and abs(ylist[i] - ylist[i-1]) > 10:  
                print(f"Sudden large change in speed value detected: {ylist[i-1]} to {ylist[i]}. Setting to previous valid value.")
                ylist[i] = ylist[i-1]

        if trackID in self.trackid2line:
            x_data, y_data, lineindx, lineh  = self.util_GetXYdata_and_LineHandle(trackID, which_axis=axis_index)
        else:
            #   Note: we age out old tracks
            self.util_RemoveStaleLines(which_axis=axis_index)

            # find first empty line handle slot
#            print("updateSpeedPlot: finding first empty line slot...")
            x_data, y_data, lineindx, lineh = self.util_FindFirstEmptyLineSlot(trackID, line_color=colordef, which_axis=axis_index)
        
        #print(f'updateSpeedPlot (AFTER CREATION): axis=0: {lineindx}, {len(x_data)}, {len(y_data)}')
        update_data = self.trackid2line[trackID]
        
        #print(f'updateSpeedPlot (BEFORE EXTEND): axis=0: {lineindx}, {len(x_data)}, {len(y_data)}, xlist length={len(xlist)}, ylist length={len(ylist)}')
        x_data.extend(xlist)
        y_data.extend(ylist)
        #print(f'updateSpeedPlot (AFTER EXTEND): axis=0: {lineindx}, {len(x_data)}, {len(y_data)}, xlist length={len(xlist)}, ylist length={len(ylist)}')
        
        #print(f"{trackID}: {lineindx}, {len(xlist)},{len(ylist)}, {len(x_data)}, {len(y_data)}" )
        #self.ax.plot(X, Y, marker='o', markersize=2, color='red')
        #ax.grid()
        #for x_i, y_i in zip(xlist, ylist):
        #    print(f"{trackID}: {lineindx}, {x_i},{y_i}" )
 
        update_data = self.trackid2line[trackID]
        update_data[axis_index] = [ lineindx, x_data, y_data ]
        self.trackid2line[trackID] = update_data
        if not self.line_removed[axis_index][lineindx] and len(x_data) > ignorelength:
            lineh.set_xdata(x_data)
            lineh.set_ydata(y_data)

        #self.ax.relim()
        ax.autoscale_view()

        # render the updated plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # ////////////////////////////////////////////////////////////////////////////////////
    # //                                                                                //
    # //  updateHeadingPlot                                                             //
    # //                                                                                //
    # ////////////////////////////////////////////////////////////////////////////////////
    def updateHeadingPlot(self,trackID, X, Y, ignorelength=0, colordef=(-1,-1,-1)):
        
        # update the line associated with the track
        axis_index = 1
        ax = self.ax[axis_index]
        ax.grid(True)

        xlist = self.util_ConvertToList(X)
        ylist = self.util_ConvertToList(Y)

        #print(f'updateHeadingPlot: trackID={trackID}, new lengths of X={len(xlist)}, Y={len(ylist)}')
        
        xlist = [0 if x < 0 else x for x in xlist ]

        # Sanity check
        # the method here can be done in real-time as the heading values are being generated, 
        # but for simplicity we do it here before plotting. We check if the heading value is 
        # outside the valid range of -180 to 180 degrees, and if so, we replace it with the 
        # last valid heading value or 0 if no valid value exists yet. This ensures that the 
        # plot will not have any invalid heading values that could distort the visualization.
        angle_last = ylist[0] if len(ylist) > 0 else 0
        for i, angle in enumerate(ylist):
            if angle < -180 or angle > 180:
                ylist[i] = angle_last
                print(f"Invalid heading value: {angle}. Setting to previous valid value or 0 if no previous valid value exists.")
            else:
                angle_last = angle
    
        # Sanity check: check for sudden large changes in heading values that are not physically plausible
        # This is a simple way to handle potential outliers or noise in the heading data that could be due to sensor errors or other issues. 
        # The threshold of 90 degrees is just an example and can be adjusted based on the expected dynamics of the tracked objects.
        # in real-time, this check can be done as each new heading value is generated, but for simplicity we do it here before plotting.
        last_est_value=ylist[0]
        for i in range(1, len(ylist)):
            ytmp = ylist[i]
            if abs(ylist[i] - last_est_value) > 90 and abs(ylist[i] < 175.) and abs(ylist[i-1] < 175.):  # Assuming a sudden change of more than 90 degrees in 0.1 sec is not plausible
                print(f"Sudden large change in heading value detected: {ylist[i-1]} to {ylist[i]}. Setting to previous valid value.")
                ylist[i] = ylist[i-1]
            last_est_value=ytmp
        
        if trackID in self.trackid2line:
            #print("updateHeadingPlot: getting existing empty line slot...")
            x_data, y_data, lineindx, lineh = self.util_GetXYdata_and_LineHandle(trackID, which_axis=axis_index)
        else:
        # add new track. 
            #   Note: we age out old tracks
            self.util_RemoveStaleLines(which_axis=axis_index)

            # find first empty line handle slot
            #print("updateHeadingPlot: finding first empty line slot...")
            x_data, y_data, lineindx, lineh = self.util_FindFirstEmptyLineSlot(trackID, line_color=colordef, which_axis=axis_index, preserve_color=True)
        
        x_data.extend(xlist)
        y_data.extend(ylist)

        #print(f"{trackID}: {lineindx}, {len(xlist)},{len(ylist)}, {len(x_data)}, {len(y_data)}" )
        #self.ax.plot(X, Y, marker='o', markersize=2, color='red')
        update_data = self.trackid2line[trackID]
        update_data[axis_index] = [ lineindx, x_data, y_data ]
        self.trackid2line[trackID] = update_data
        if not self.line_removed[axis_index][lineindx] and len(x_data) > ignorelength:
            lineh.set_xdata(x_data)
            lineh.set_ydata(y_data)

        #self.ax.relim()
        ax.autoscale_view()

        # render the updated plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # ////////////////////////////////////////////////////////////////////////////////////
    # //                                                                                //
    # //  util_GetXYdata_and_LineHandle                                                 //
    # //                                                                                //
    # ////////////////////////////////////////////////////////////////////////////////////
    def util_GetXYdata_and_LineHandle(self, trackID, which_axis):
        data = self.trackid2line[trackID]
        dataset = data[which_axis]
        lineindx = dataset[0]
        x_data = dataset[1]
        y_data = dataset[2]
        lineh = self.lines[which_axis][lineindx]
        return x_data, y_data, lineindx, lineh

    # ////////////////////////////////////////////////////////////////////////////////////
    # //                                                                                //
    # //  util_RemoveStaleLines                                                         //
    # //                                                                                //
    # ////////////////////////////////////////////////////////////////////////////////////
    def util_RemoveStaleLines(self, which_axis):
        if len(self.trackid2line) >= 50:
#           self.trackid2line.pop(list(self.trackid2line.keys())[0])
            for tID in list(self.trackid2line):
                data = self.trackid2line.pop(tID)
                lineindex = data[which_axis][0]
                break
            lineh = self.lines[which_axis][lineindex]
            lineh.remove()
            self.line_removed[which_axis][lineindex] = True

    # ////////////////////////////////////////////////////////////////////////////////////
    # //                                                                                //
    # //  util_FindFirstEmptyLineSlot                                                   //
    # //  Creates a new data and line handles for all sub-plots                         //
    # //                                                                                //
    # ////////////////////////////////////////////////////////////////////////////////////
    def util_FindFirstEmptyLineSlot(self, trackID_IN, which_axis, preserve_color=False, line_color=None):
        for lineindx in range(len(self.line_removed[which_axis])): # same number of lines for both subplots
            if self.line_removed[0][lineindx] or self.line_removed[1][lineindx]:
                x_data1 = []
                y_data1 = []
                x_data2 = []
                y_data2 = []
                data_init = [ [int, None, None], [int, None, None] ]
                data_init[0] = [ lineindx, x_data1, y_data1 ]
                data_init[1] = [ lineindx, x_data2, y_data2 ]
                self.trackid2line[trackID_IN] = data_init

                # create a new line for all subplots, with same color as first one
                for axis in range(len(self.ax)):
                    self.lines[axis][lineindx], = self.ax[axis].plot( [], [] )
                    self.line_removed[axis][lineindx] = False
                    lineh = self.lines[axis][lineindx]
                    self.setup_linestyle(lineh, color_IN=line_color, use_last_color=True if axis > 0 else False) 
                break

        lineh = self.lines[which_axis][lineindx]
        x_data = self.trackid2line[trackID_IN][which_axis][1]
        y_data = self.trackid2line[trackID_IN][which_axis][2]
        return x_data, y_data, lineindx, lineh
    
    # ////////////////////////////////////////////////////////////////////////////////////
    # //                                                                                //
    # //  util_ConvertToList                                                            //
    # //                                                                                //
    # ////////////////////////////////////////////////////////////////////////////////////
    def util_ConvertToList(self, X):
        xlist = []
        if type(X) is list:
            xlist.extend(X)
        elif type(X) is np.ndarray:
            xlist.extend(X.tolist())
        elif type(X) is float or type(X) is int:
            xlist.append(X)
        elif type(X) is np.float64 or type(X) is np.int64 or type(X) is np.float32 or type(X) is np.int32 or type(X) is np.float32:
            xlist.append(float(X))
        return xlist