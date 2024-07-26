# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:07:25 2024

@author: owena
"""

#import pyPDAF.PDAF as PDAF
import numpy as np
from observations import PointObservation, DictObsReader

def create_obs_from_truth(truth,indt,ind0,ind1,sigo):
    """ 
    Create observation from artificial truth. 
    
    Parameters
    ----------
    truth : models.Model object 
        Truth model after completion of its model run. 
    indt : list of int 
        Time indices for which observations are created. 
    ind0,iind1 : list of int 
        Observations are made at points (i0,i1) with i0 in 
        ind0 and i1 in ind1 for every combo of i0,i1.
    sigo : float 
        Observational error standard deviation scale factor. 
    
    Returns
    -------
    obs : observations.PointObservation 
        Observation operator associated with these observations. 
    """
    ocoords   = np.array([(it,i0,i1) for it in indt
                          for i0 in ind0
                          for i1 in ind1])
    observed  = np.array([truth.saved_output[it,i0,i1] for it,i0,i1 in ocoords])
    observed += sigo*np.random.normal(size=np.shape(observed))*observed
    ocoords = np.array(ocoords, dtype=float)
    reader = DictObsReader(0,
                           {'observed': observed,
                            'coord':ocoords[:,1:],'time':ocoords[:,0]*truth.dt,
                            'variance':sigo**2*np.ones_like(observed)})

    obs = PointObservation(0, 0, reader, truth.nn_interpolator)
    obs.create_windows_from_model(truth.time_init, truth.save_steps[-1], truth.dt)
    print(reader.data)
    return obs