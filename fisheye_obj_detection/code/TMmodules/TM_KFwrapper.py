##########################################################################
#                                                                        #
#  © 2026 Regents of the University of Minnesota. All rights reserved.   #
# This program is shared under the terms and conditions of the GNU       #
# Affero General Public License 3.0, License. Further details about      #
# the GNU Affero GPL 3.0 license are available in the LICENSE text file. #
#                                                                        #
##########################################################################

###############################################################################
##                                                                           ##
##                       Kalman filter wrapper                               ##
##   Initiates error cov. priors based on class                              ##
##                                                                           ##
##   To perform predictions and updates you need to assign a numeric         ##
##   track ID. Then the app can check what was assigned to make sure the     ##
##   correct KF object is being associated with the desired track ID         ##
###############################################################################

import numpy as np
import sys, io, math
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class TM_KFwrapper:

    valid_filter=True
    Pcov = None
    Qn = None
    Rmeas = None
    instcnt=0
    trackID=-1
    dt =1.0
    state_dim=None
    def __init__(self, state_dim, measure_dim=2, dt=1.0, detclass='car', trackID_IN=-1 ):
        if state_dim != 4 and state_dim != 6:
            self.valid_filter=False
            print('TM_KFwrapper ERROR: The dimensions must be either 4 or 6!\n')
            sys.exit(1)
        
        if (self.trackID != -1 ):
            return
        
        self.assignTrackID(trackID_IN)
        self.kf = KalmanFilter(dim_x=state_dim, dim_z=measure_dim)
        Pcov = np.eye(state_dim)
        Qn   = np.eye(state_dim)
        Rmeas = np.eye(measure_dim)
        Qn *= 30.0 # *= 15.0
        self.state_dim = state_dim
        #Qn[2:4,2:4] *=25
         
        if state_dim == 4:
            self.kf.F = np.array([[1., 0., dt, 0.],
                             [0., 1., 0., dt],
                             [0., 0., 1., 0.],
                            [0., 0., 0., 1.]])
            self.kf.H = np.array( [[1., 0., 0., 0.],
                                   [0., 1., 0., 0.]])
            self.kf.x = np.array([0.,0.,0.,0.])
        

        else:  # include acceleration
            self.kf.F = np.array([[1., 0., dt, 0., dt*dt, 0.],
                                  [0., 1., 0., dt, 0.,    dt*dt],
                                  [0., 0., 1., 0., dt,    0.],
                                  [0., 0., 0., 1., 0.,    dt],
                                  [0., 0., 0., 0., 1., 0.],
                                  [0., 0., 0., 0., 0., 1.]])
            self.kf.x = np.array([0.,0.,0.,0.,0.,0.])
            self.kf.H = np.array( [[1., 0., 0., 0., 0., 0.],
                                   [0., 1., 0., 0., 0., 0.]])

        # to do: tuning per detected object class
        if detclass=='car' or detclass=='pickup' or detclass=='van' or detclass=='cargo van':
            xy_var = 10.0
            xyvel_var = 1000.0
            xyaccn_var = 10000.0

        elif detclass=='truck' or detclass=='bus':
            xy_var = 10.0
            xyvel_var = 1000.0
            xyaccn_var = 10000.0

        elif detclass=='person':
            xy_var = 8.0
            xyvel_var = 64.0
            xyaccn_var = 900.0

        elif detclass=='motorcycle' or detclass=='bicycle':
            xy_var = 10.0
            xyvel_var = 1000.0
            xyaccn_var = 1000.0

        else:
            self.valid_filter=False
            print(f'TM_KFwrapper ERROR: class name: {detclass} is not recognized!\n')
            print(f'  The choices are: person, car, truck, bus, bicycle, motorcycle, bus, cargo van.')
            return

        Pcov[0:2,0:2] *= xy_var
        Pcov[2:4,2:4] *= xyvel_var
        if state_dim == 6:
            Pcov[4:6,4:6] == xyaccn_var
        Rmeas *= xy_var*10 # observation is worse than state cov.

        self.kf.R  = Rmeas
        self.kf.P = Pcov
        self.kf.Q = Qn
    
    def setdt(self, dt):
        assert dt > 0, "TM_KFwrapper.setdt: input parameter, deltaT must be > 0!"
        self.deltaT=dt
        if self.state_dim==6:
            self.kf.F = np.array([[1., 0., dt, 0., dt*dt, 0.],
                              [0., 1., 0., dt, 0.,    dt*dt],
                              [0., 0., 1., 0., dt,    0.],
                                  [0., 0., 0., 1., 0.,    dt],
                                  [0., 0., 0., 0., 1., 0.],
                                  [0., 0., 0., 0., 0., 1.]])
        else:
            self.kf.F = np.array([[1., 0., dt, 0.],
                             [0., 1., 0., dt],
                             [0., 0., 1., 0.],
                             [0., 0., 0., 1.]])
         
    def assignTrackID(self, id):
        if id != self.trackID:
            self.trackID = id
            self.instcnt=0

    def resetTrackID(self):
        self.trackID = -1

    def init_state(self, XYp ):
        if not self.valid_filter or self.trackID < 0:
            return
        self.kf.x[0] = XYp[0]
        self.kf.x[1] = XYp[1]

    def get_state(self):
        if not self.valid_filter or self.trackID < 0:
            return [],0
        return(self.kf.x, self.instcnt)
    
    def predict(self):
        if not self.valid_filter or self.trackID < 0:
            return
        self.kf.predict()
        #print('\nP-apriori, x-predict=\n', self.kf.P_prior, self.kf.x_prior)
        #print('\nP-posterior=\n', self.kf.P_post)

    # state covariance is not updated.
    def predict_steadystate(self):
        if not self.valid_filter or self.trackID < 0:
            return
        self.kf.predict_steadystate()
        
    # returns a-priori state prediction
    def get_prediction(self):
        if not self.valid_filter or self.trackID < 0:
            return [],0
        return(self.kf.x_prior, self.instcnt)
    
    def getTrackID(self):
        return self.trackID
    
    def update_measures(self, XYp ):
        if not self.valid_filter or self.trackID < 0:
            return
        self.instcnt += 1  # increment for every new track pos measurement 
        self.kf.update(np.asarray(XYp))
        #self.kf.update(XYp)
        #print(f'\nID={self.trackID}:\n{self.kf.K}')
             
    # return the full state estimation after an update
    def get_estimate(self):
        if not self.valid_filter or self.trackID < 0:
            return [],0
        x_post = self.kf.x_post.reshape(self.state_dim,)
        #print(f"x_post ID={self.trackID}: {x_post}, {self.kf.x}")
        return( self.kf.x, self.instcnt)
