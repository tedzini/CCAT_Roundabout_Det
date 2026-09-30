##########################################################################
#                                                                        #
#  © 2026 Regents of the University of Minnesota. All rights reserved.   #
# This program is shared under the terms and conditions of the GNU       #
# Affero General Public License 3.0, License. Further details about      #
# the GNU Affero GPL 3.0 license are available in the LICENSE text file. #
#                                                                        #
##########################################################################

import sys
import numpy as np

# todo: generalize to handle any vector
class TM_mean_cov_update:
    def __init__(self, mean0=0, cov0=0, cnt0=0):
        self.mn = mean0
        self.cov = cov0
        self.ncnt = cnt0
        return
    
    def update(self, Xnew ):

        # work with numpy arrays        
        if type(Xnew) != np.ndarray and type(Xnew) != list:
            Xnew = [Xnew]
            Xarr = np.array(Xnew, dtype=np.float64)
        else:
            Xarr = np.array(Xnew, dtype=np.float64)

        self.vecdim = Xarr.shape[0]
        Xarr = Xarr[:, np.newaxis] # make it a column vector
        
        if self.ncnt == 0:
            self.mn = Xarr
            self.cov = np.zeros((self.vecdim, self.vecdim), dtype=np.float64)
            self.ncnt = 1
            return
        
        mn_last = self.mn.copy() # make a copy to avoid modifying the original during calculations
        inv_Nplus1 = 1./(self.ncnt+1)
        N = self.ncnt

        # update mean
        mn =  inv_Nplus1 * (N*mn_last + Xarr)
     
        # update covariance
        cov_last = self.cov.copy() # make a copy to avoid modifying the original during calculations
        cov = inv_Nplus1 * (N*cov_last + N*mn_last @ mn_last.T + Xarr @ Xarr.T)
        cov = cov - inv_Nplus1*inv_Nplus1*(N*mn_last + Xarr) @ (N*mn_last + Xarr).T

        self.cov = cov
        self.mn = mn
        self.ncnt += 1

    def Mean(self):
        return self.mn.tolist()
    
    def Covariance(self):
        return self.cov.tolist()
    
    def stdev(self, diag=False):
        # Compute eigenvalues and eigenvectors
        vals, vecs = np.linalg.eigh(self.cov)

        # Reconstruct square root matrix: V * sqrt(D) * V.T
        # Use np.maximum to handle tiny negative values due to precision errors
        if diag:
            sqrt_matrix = np.diag(np.sqrt(np.maximum(vals, 0)))
        else:
            sqrt_matrix = vecs @ np.diag(np.sqrt(np.maximum(vals, 0))) @ vecs.T
        return sqrt_matrix.tolist()

    def Count(self):
        return self.ncnt
    

    

