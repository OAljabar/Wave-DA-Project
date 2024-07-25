"""This file is part of pyPDAF

Copyright (C) 2022 University of Reading and
National Centre for Earth Observation

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from abc import ABC, abstractmethod
import numpy as np 
from general import check_init
import xarray as xr

class Model(ABC):
    """ 
    Abstract class for a PDAF compatible model. 
    
    Methods
    -------
    init_fields 
        Initialises model field at beginning of model run. 
    step_forward
        Step the model forward in time from given step.  
    collect_state_pdaf
        Outputs local model fields as array to be collected by PDAF.
    distribute_state_pdaf
        Accepts corrected local model fields from PDAF as 1D array.
    dim_state : int 
        Size of model state. 
    dim_state_p : int 
        Size of model state on local process.
    time : float 
        Current model time.  
        
    Attributes
    ----------
    dt : float>0
        Numerical time step.
    step : int>=0
        Current model step. 
    step_init : int>=0
        Step at which model is initialized. 
    time_init : float 
        Time at which model is initialized. 
    control : parallelization.ProcessControl 
        Object controlling parallelization. 
    
    """
    
    @property
    @abstractmethod
    def dim_state(self):
        """Return size of total model state."""
        
    @property
    def dim_state_p(self):
        """
        Return size of model state on this process.
        
        By default is assumed that only 1 process is used.
        """
        return self.dim_state
    
    @property 
    def time(self):
        """ Return current model time interval. """
        return self.step2time(self.step)
    
    def step2time(self, step):
        """ Return time for step.  """
        return float(self.tinit + step*self.dt)
    
    def time2step(self, time):
        """ Return step associated with time. """
        return int((time - self.tinit)/self.dt)
    
    @abstractmethod
    @check_init
    def init_fields(self, process_control):
        """
        Initialises model field at beginning of model run. 
        
        After this method step_init and time_init must be set. 
        
        Parameters
        ----------
        process_control : parallelization.ProcessControl 
            Object containing information about processes used by this model. 
        """
    
    @abstractmethod 
    def step_forward(self, step, steps_forward):
        """ 
        Step the model forward in time from given step.  
        
        Parameters
        ----------
        step : int>=0
            Current time step.
        steps_forward : int>0
            Number of steps        
            
        """
    @abstractmethod
    def collect_state_pdaf(self, dim_p, state_p):
        """Outputs local model fields as array to be collected by PDAF.

        Aim of this method is to reshape the different models fields 
        into 1 long 1D array that and returns this one so it can 
        be processed by PDAF. As such it is the opposite of 
        `distribute_state_pdaf`. Relies on Python's pass by reference. 
        The interface of this method should not be changed as it must 
        match the equivalent PDAF interface. 

        Parameters
        ----------
        dim_p : int 
            Size of array state_p.
        state_p : ndarray
            Allocated 1D array to store the output of this method.    
            
        Returns
        -------
        state_p : 1D numpy array 
            Model fields for this process concatenated into 1D array 
            in Fortran ordering.  
    
    """
    
    @abstractmethod
    def distribute_state_pdaf(self, dim_p, state_p):
        """Accepts corrected local model fields from PDAF as 1D array. 

        Aim of this method is to reshape the 1D array from PDAF back
        into different model fields of different sizes. As such 
        it is the opposite to `collect_state_pdaf`. Relies on Python's 
        pass by reference. The interface of this method should not be 
        changed as it must match the equivalent PDAF interface. 

        Parameters
        ----------
        dim_p : int 
            Size of state_p. 
        state_p : 1D numpy array
            1D array containing the model fields for this process.
        
        Returns
        -------
        state_p :
            Same as intput state_p

        """
        
class FugroModel(Model):
    """ 
    Example of a very simple model. 
    
    Attributes 
    ----------
    dt : float 
        Time step
    shape : (2,) tuplet of int
        Size of the model field. 
    var : float 
        Variance AR1 process. 
    corr : float 
        Correlation AR1 process spaced 1 spatial step apart. 
    seed : int 
        Seed for random number generation. 
    is_initialized : bool
        Flag indicating whether fields were created. 
        
    Methods 
    -------
    nn_interpolate
        Given grid index coordinates return 
        indices in state_p/weights to carry out 
        nearest neighbor interpolation. 
        
    """
    
    def __init__(self, member, tinit, tfinal, filedir,savedir,save_states = False,\
                 finedomain = False,variables = ['hs','t01','t02'],label = 'da'):
        self.dt = 1.
        #self.var = (np.sqrt(var * (1-corr**2)), corr)
        self.member = member
        self.tinit = tinit
        self.tfinal = tfinal
        self.finedomain = finedomain
        self.variables = variables
        self.nvars = len(self.variables)
        self.save_states = save_states
        self.label = label
        self.savedir = savedir
        
        if finedomain:
            self.filefmt = filedir + '{:02d}/netcdf/ww3.sgom.202203{{:02}}12.nc'\
                .format(member)
        else:
            self.filefmt = filedir + '{:02d}/netcdf/ww3.gom.202203{{:02}}12.nc'\
                .format(member)
       
    @property 
    def dim_state(self):
        return np.prod(self.shape)*self.nvars
        
    def init_fields(self, process_control):
        """
        Loads model fields for all runs
        """
        
        self.control = process_control
        
        data = xr.open_dataset(self.filefmt.format(int(self.tinit)))[self.variables]\
            .astype('float16')
        data = data.rename({'time':'leadtime'})
        data = data.assign_coords(leadtime = np.arange(len(data.leadtime))/24)
        
        for run in range(int(self.tinit+1),int(self.tfinal+1)):
            
            filepath = self.filefmt.format(run)
            nextdata = xr.open_dataset(filepath)[self.variables].\
                astype('float16')
            nextdata = nextdata.rename({'time':'leadtime'})
            nextdata = nextdata.assign_coords(leadtime = np.arange(len(nextdata.leadtime))/24)
            data = xr.concat([data,nextdata],dim = 'step')
        
        self.fields = data
        self.dsattrs = {variable:self.fields[variable].attrs\
                        for variable in self.variables}
        self.step = 0
        
        self.state = self.fields.sel(step = self.step)
        self.shape = np.shape(self.state[self.variables[0]])
        
        self.is_initialized = True
        
        return self.state
      
    @check_init      
    def step_forward(self, step, steps_forward):
        
        '''
        Steps the model forward by specified number of steps from a specified step
        '''
        
        self.step = step + steps_forward
        self.state = self.fields.sel(step = self.step)
        
    def collect_state_pdaf(self, dim_p, state_p):
        
        """Outputs model fields as array to be collected by PDAF.

        Parameters
        ----------
        dim_p : int 
            Size of array state_p.
        state_p : ndarray
            Allocated 1D array to store the output of this method.    
            
        Returns
        -------
        state_p : 1D numpy array 
            Model fields for this process concatenated into 1D array 
            in Fortran ordering.  
        """
        
        hs = np.reshape(self.state['hs'].data,dim_p//self.nvars,order = 'F')
        tm01 = np.reshape(self.state['t01'].data,dim_p//self.nvars,order = 'F')
        tm02 = np.reshape(self.state['t02'].data,dim_p//self.nvars,order = 'F')
        state_p = np.concatenate((hs,tm01,tm02))
        
        return state_p
        
    def distribute_state_pdaf(self, dim_p, state_p):
        
        """Accepts corrected local model fields from PDAF as 1D array. 

        Aim of this method is to reshape the 1D array from PDAF back
        into different model fields of different sizes. As such 
        it is the opposite to `collect_state_pdaf`. Relies on Python's 
        pass by reference. The interface of this method should not be 
        changed as it must match the equivalent PDAF interface. 

        Parameters
        ----------
        dim_p : int 
            Size of state_p. 
        state_p : 1D numpy array
            1D array containing the model fields for this process.
        
        Returns
        -------
        state_p :
            Same as intput state_p

        """
        
        hs = np.reshape(state_p[:dim_p//self.nvars],self.shape,order = 'F')
        tm01 = np.reshape(state_p[dim_p//self.nvars:2*dim_p//self.nvars],self.shape,order = 'F')
        tm02 = np.reshape(state_p[2*dim_p//self.nvars:],self.shape,order = 'F')
        
        data = xr.Dataset(data_vars = {'hs':(['leadtime','latitude','longitude'],hs,self.dsattrs['hs']),\
                                          't01':(['leadtime','latitude','longitude'],tm01,self.dsattrs['t01']),\
                                          't02':(['leadtime','latitude','longitude'],tm02,self.dsattrs['t02'])},\
                             coords = {'latitude':self.fields.latitude,\
                                       'longitude':self.fields.longitude,\
                                       'leadtime':self.fields.leadtime})
        
        self.state = data
        if self.save_states:
            self.write_output()
            # for variable in self.variables:
            #     self.fields[variable][self.step] = self.state[variable]
        
        return state_p
    
    def nn_interpolator(self,variable):
        
        """
        Returns interpolator to be used by PDAF

        Parameters
        ----------
        variable : str
            variable to find indices for

        Returns
        -------
        interpolator : function
            Function converting leadtime,lat,lon indices to indices in state
            vector and weights.

        """
        
        try:
            varindex = self.variables.index(variable)
        except:
            raise Exception('Invalid variable selected. Valid variables are ' + \
                            self.variables)
                
        def interpolator(coords):
            
            '''
            Interpolator which accepts leadtime,lat,lon indices and returns corresponding
            indices in state vector along with weights for assimilation
            
            Parameters
            ----------
            coords : ndarray of float
                n*3 array containing leadtime, lat, lon coords for each point
            
            Returns
            -------
            indices : ndarray of int
                n*1 array containing index in state vector of each point
            weights : ndarray of float
                n*1 array containing weights to be given to each observation
            '''
        
            # convert to coordinates in state vector
            indices = np.ravel_multi_index(coords.astype('int').T, dims=self.shape,order = 'F')
            indices = np.reshape(indices, (-1,1))
            weights = np.ones_like(indices, dtype=float)
            weights = np.where(np.logical_and(indices>=0, indices<self.dim_state),
                               weights, 0.0)
            
            indices += varindex*self.dim_state//self.nvars
            
            return indices, weights
        
        return interpolator
  
    
    def write_output(self,savedir = None):
        
        """
        Writes current state to netcdf file
        
        Parameters
        ----------
        savedir : str
            Directory to save output in
        """
        
        if savedir is None:
            savedir = self.savedir
        
        filename = savedir + 'step{:02d}member{:02d}'.format(int(self.step),self.member)\
            +self.label+'.nc'

        self.state.astype('f8').to_netcdf(filename)
        print(f'Saved current state at {self.step} at '+ filename)
        
        
