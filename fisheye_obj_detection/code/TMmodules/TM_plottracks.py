##########################################################################
#                                                                        #
#  © 2026 Regents of the University of Minnesota. All rights reserved.   #
# This program is shared under the terms and conditions of the GNU       #
# Affero General Public License 3.0, License. Further details about      #
# the GNU Affero GPL 3.0 license are available in the LICENSE text file. #
#                                                                        #
##########################################################################

import math
import numpy as np
import numpy.random as npr
import matplotlib
# note: on windows platforms, interactive fig. windows shuts down when trying to move or resize them.
# This has been a reported issue with matplotlib that happens here. One suggested workaround is to use the 
# 'TkAgg' or 'Qt5Agg' backend. This still didn't resolve the issues here, but it may be worth trying on other platforms.
# matplotlib.use('TkAgg') # 'TkAgg' or 'Qt5Agg'
import matplotlib.pyplot as plt
from   matplotlib.patches import Polygon

# ////////////////////////////////////////////////////////////////////////////////////
# //                                                                                //
# //  plot vehicle tracks is state plane coords.                                    //
# //                                                                                //
# ////////////////////////////////////////////////////////////////////////////////////
class TM_plottracks:
    fig=None
    ax=None
    lines=[None]*50
    trackid2line = {}
    line_removed=[True]*50
    line_cnt=0
    cnt=-150
    plot_heading = False
    plot_speed = False
    texth = []

    def __init__(self, EW_Range, NS_Range, minRad, skirtRad, outRad ):

        # sanity check
        assert EW_Range[0] < EW_Range[1], "Easterling max must be greater than min value!"
        assert NS_Range[0] < NS_Range[1], "Northing max must be greater than min value!"
        plt.ion()
        # use fig, axs = plt.subplots(rows, cols, figsize = (5,5)) for multiple subplots
        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(6,6)
        self.ax.set_aspect('equal', adjustable='box')
#        for n in range(len(self.lines)):
#            line, = self.ax.plot( [], [] )
#           self.lines[n] = line
#            self.line_removed[n] = False
        self.min_X = EW_Range[0]
        self.max_X = EW_Range[1]
        self.min_Y = NS_Range[0]
        self.max_Y = NS_Range[1]
        self.ax.set_xlim(self.min_X,self.max_X)
        self.ax.set_ylim(self.min_Y,self.max_Y)
        plt.pause(0.1) 
        plt.show()

        plt.title("Intersection Tracks")
        plt.xlabel("Easting (ft.)")
        plt.ylabel("Northing (ft.)")

        # plot the roundabout
        Xpnt = [0.0]*360
        Ypnt = [0.0]*360
        self.ux = [0.0]*360
        self.uy = [0.0]*360
        for angle in range(360):
            self.ux[angle] = math.cos(math.radians(angle))
            self.uy[angle] = math.sin(math.radians(angle))
            
        for angle in range(360):
            Xpnt[angle] = self.ux[angle] * outRad
            Ypnt[angle] = self.uy[angle] * outRad

        circle_outer, = self.ax.plot( [], [] )
        circle_outer.set_xdata( Xpnt )
        circle_outer.set_ydata( Ypnt )
        circle_outer.set_linestyle('--')
        circle_outer.set_linewidth(1)
        circle_outer.set_color('black')
                
        verticies_skirt = []
        for n in range(360):
            verticies_skirt.append( [self.ux[n] * skirtRad, self.uy[n] * skirtRad ] )
        polygon_skirt = Polygon(verticies_skirt, closed=True, facecolor=(193./255., 154./255., 107./255.), edgecolor=None)
        self.ax.add_patch(polygon_skirt)
         
        verticies = []
        for n in range(360):
            verticies.append( [self.ux[n] * minRad, self.uy[n] * minRad ] )
            
        polygon = Polygon(verticies, closed=True, facecolor=(0.05,0.6,0.05), edgecolor=None)
        self.ax.add_patch(polygon)
        self.ax.grid(visible=True, linestyle='--', linewidth=0.5, color='gray', axis='both')
    
    # ////////////////////////////////////////////////////////////////////////////////////
    # //                                                                                //
    # //  plot vehicle tracks is state plane coords.                                    //
    # //                                                                                //
    # ////////////////////////////////////////////////////////////////////////////////////
    def setup_linestyle(self, lineh, color_IN=(-1,-1,-1), markersize=2):
        lineh.set_linewidth(1)
        lineh.set_linestyle('none')
        lineh.set_marker('o')
        if color_IN[0] < 0 or color_IN[1] < 0 or color_IN[2] < 0:
            color_IN = list(np.random.rand(3))
        #print(color_IN)
        lineh.set_color(color_IN)
        lineh.set_markerfacecolor(color_IN)
        lineh.set_markersize(markersize)

    # ////////////////////////////////////////////////////////////////////////////////////
    # //                                                                                //
    # //  plot vehicle tracks is state plane coords.                                    //
    # //                                                                                //
    # ////////////////////////////////////////////////////////////////////////////////////
    def updatePlot(self,trackID, X, Y, colordef=(-1,-1,-1), ignorelength=0):
        
        # update the line associated with the track
        if trackID in self.trackid2line:
            data = self.trackid2line[trackID]
            lineindx = data[0]
            x_data = data[1]
            y_data = data[2]
            lineh = self.lines[lineindx]
        else:
        # add new track. Note: we age out old tracks
            if len(self.trackid2line) >= 50:
#                self.trackid2line.pop(list(self.trackid2line.keys())[0])
                for tID in list(self.trackid2line):
                    data = self.trackid2line.pop(tID)
                    lineindex = data[0]
                    break
                self.lines[lineindex].remove()
                #self.fig.canvas.draw()
                self.line_removed[lineindex] = True

            # find first empty line handle slot
            for lineindx in range(len(self.line_removed)):
                if self.line_removed[lineindx]:
                    x_data = []
                    y_data = []
                    data = [ lineindx, x_data, y_data ]
                    self.trackid2line[trackID] = data
                    # init with empty line data
                    self.lines[lineindx], = self.ax.plot( [], [] )
                    self.line_removed[lineindx] = False
                    lineh = self.lines[lineindx]
                    self.setup_linestyle(lineh, colordef) 
                    break
        
        if type(X) is list:
            x_data.extend(X)
            y_data.extend(Y)
        elif type(X) is float or type(X) is int:
            x_data.append(X)
            y_data.append(Y)
            
        #self.ax.plot(X, Y, marker='o', markersize=2, color='red')
        #print(f"{trackID}: {lineindx}, {X},{Y}, {len(x_data)}, {len(y_data)}" )
        update_data = [ lineindx, x_data, y_data ]
        self.trackid2line[trackID] = update_data
        if not self.line_removed[lineindx] and len(x_data) > ignorelength:
            lineh.set_xdata(x_data)
            lineh.set_ydata(y_data)

        #self.ax.relim()
        self.ax.autoscale_view()

        # render the updated plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        

    # ////////////////////////////////////////////////////////////////////////////////////
    # //                                                                                //
    # //                                                                                //
    # ////////////////////////////////////////////////////////////////////////////////////
    def plottext(self, x,y,text:str, fgcolor='white', bgcolor='black', fontsize=12):
        texth_l = self.ax.text( x,y, s=text, 
                            color=fgcolor, 
                            weight='bold', 
                            fontsize=fontsize,
                            bbox=dict(facecolor=bgcolor, alpha=0.9, pad=4))
        self.texth.append(texth_l)

    # ////////////////////////////////////////////////////////////////////////////////////
    # //                                                                                //
    # //                                                                                //
    # ////////////////////////////////////////////////////////////////////////////////////
    def removeText(self):
        if len(self.texth)==0:
            return
        
        for text_hdl in self.texth:
            text_hdl.remove()
        self.texth=[]
        
    # ////////////////////////////////////////////////////////////////////////////////////
    # //                                                                                //
    # //                                                                                //
    # ////////////////////////////////////////////////////////////////////////////////////
    def removeTrackedLine(self,dead_trackID):
        if dead_trackID in self.trackid2line:
            data =  self.trackid2line.pop(dead_trackID)
            lineindx = data[0]
            lineh = self.lines[lineindx]
            lineh.remove()
            self.line_removed[lineindx] = True
