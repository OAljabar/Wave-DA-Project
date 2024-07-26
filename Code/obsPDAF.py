# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 14:55:21 2024

@author: owena
"""

from obsdata import getobs
import numpy as np
import xarray as xr

class OBS(getobs):
    
    n_obs = 1
    
    def __init__(self, variable, mype_filter,
                 nx, doassim, delt_obs, rms_obs):
        """constructor

        Parameters
        ----------
        variable : string
            name of the observation variable
        mype_filter : int
            rank of the PE in filter communicator
        nx : ndarray
            grid size of the model domain
        doassim : int
            whether to assimilate this observation type
        delt_obs : int
            time step interval for observations
        rms_obs_scale : float
            scale of observation error standard deviation (assuming error
            scales linearly with the variable being measured)
        """
        OBS.n_obs += 1

        self.i_obs = OBS.n_obs

        assert OBS.n_obs >= 1, 'observation count must start from 1'

        if (mype_filter == 0):
            print(('Assimilate observations:', variable))
        
        self.variable = variable

        self.doassim = doassim
        self.delt_obs = delt_obs
        self.rms_obs = rms_obs

        # Specify type of distance computation
        # 0=Cartesian 1=Cartesian periodic
        self.disttype = 0

        # Number of coordinates used for distance computation
        # The distance compution starts from the first row
        self.ncoord = len(nx)

        # Allocate process-local index array
        # This array has as many rows as required
        # for the observation operator
        # 1 if observations are at grid points;
        # >1 if interpolation is required
        self.nrows = 1

        # Size of domain for periodicity for disttype=1
        # (<0 for no periodicity)
        if self.i_obs == 1:
            self.domainsize = np.zeros(self.ncoord)
            self.domainsize[0] = nx[1]
            self.domainsize[1] = nx[0]
        else:
            self.domainsize = None

        # Type of observation error: (0) Gauss, (1) Laplace
        self.obs_err_type = None

        # Whether to use (1) global full obs.
        # (0) obs. restricted to those relevant for a process domain
        self.use_global_obs = 1

        self.icoeff_p = None
    
    def getobs(self,filepath,starttime = None,endtime = None):
        
        """
        Loads observation data for specified variables between given times
        
        Parameters
        ----------
        filepath : str
            Path to text or netcdf file containing observations
        starttime : datetime object
            Time of earliest observation to load
        endtim : datetime object
            Time of latest observation to load
        variables : list of str
            Name of variables to load
        """
        
        if filepath[-3:] == 'txt':
            print('hi')
            getobs.__init__(self,filepath,[self.variable])
        else:
            self.data = xr.open_dataset(filepath)[self.variable]
            self.lat = 18.41364
            self.lon = -93.7704
            self.data = self.data.interp(latitude = self.lat,longitude = self.lon,\
                                       method = 'nearest')
        self.data = self.data.sel(time = slice(starttime,endtime))
        self.starttime = self.data.time[0].data
        self.endtime = self.data.time[-1].data
        
    
    def reader(self,window = None):
        
        """
        Returns observations from specified time window in correct format for
        DA system. If no window specified, returns array of times for which
        observations are available
        """
        
        times = self.data.time
        
        if window is None:
            return {'time':times}
        
        data = self.data.sel(time = slice(window[0],window[1]))
        times = data.time.data
        data = data.rename({'time':'leadtime'})
        data = data.assign_coords(leadtime = range(len(data.leadtime)))
        
        values = data.data
        
        tcoords = data.leadtime
        latcoords = self.lat*np.ones(len(tcoords))
        loncoords = self.lon*np.ones(len(tcoords))
        coords = np.stack((tcoords,latcoords,loncoords)).T
        
        if not hasattr(self,'ivar_obs_p'):
            self.set_ivar_obs_p()
        ivariance = self.ivar_obs_p
        
        return {'time':times,'values':values,'coords':coords,'ivariance':ivariance}
    
    def set_ivar_obs_p(self):
        
        """
        Set ivar_obs_p
        """
        
        self.ivar_obs_p = 1/(self.data*self.data*self.rms_obs*self.rms_obs)