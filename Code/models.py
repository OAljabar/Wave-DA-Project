# -*- coding: utf-8 -*-

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
import numpy as np
import copy
from abc import ABC, abstractmethod
import xarray as xr

class Model(ABC):
   """ 
   Abstract class for a PDAF compatible model. 
   
   Methods
   -------
   init_fields 
       Initialises model field at beginning of model run. 
   step 
       Step the model forward in time from given step.  
   collect_state_pdaf
       Outputs local model fields as array to be collected by PDAF.
   distribute_state_pdaf
       Accepts corrected local model fields from PDAF as 1D array. 
       
   Attributes
   ----------
   dim_state_p 
       
   """
   
   @abstractmethod
   def init_fields(self):
       """
       Initialises model field at beginning of model run. 
       """
   
   @abstractmethod 
   def step(self, step):
       """ 
       Step the model forward in time from given step.  
       
       Parameters
       ----------
       step : int
           current time step
       use_pdaf : bool
           whether PDAF is used at this step
           
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
    
    def __init__(self,member,finedomain = False,\
                 variables = ['hs','t01','t02'],
                 filedir = '/storage/silver/metstudent/msc/users_2024/mg838076/data/'):
        
        """
        Creates model object storing fields to be corrected from a given model
        run
        
        Parameters
        ----------
        member : int
            Ensemble member to be used
        finedomain : bool
            Specifies whether to use data from inner or outer domain
        filefmt : str, optional
            Format string pointing to the location in storage of the files
            containing model fields
        
        Attributes
        ----------
        filepath : str
            String pointing to the location of the specific file to be used
        runstart : int
            Start date of model run to be used
        member : int
            Ensemble member to be used
        starttime : int
            Timestep to begin the assimilation from
        time : int
            Curent timestep
        fields : xarray DataSet
            DataSet containing the model fields for all timesteps
        state : xarray DataSet
            Tracks model state at current timestep
        """
        if finedomain:
            self.filefmt = filedir + '/{:02d}/netcdf/ww3.sgom.202203{{:02}}12.nc'\
                .format(member)
        else:
            self.filefmt = filedir + '/{:02d}/netcdf/ww3.gom.202203{{:02}}12.nc'\
                .format(member)
        
        self.member = member
        self.variables = variables
        self.nvars = len(variables)
        self.save_states = False
        
    def init_fields(self,runstart_init,runstart_final):
        
        """
        Loads model fields for all runs
        
        """
        
        data = xr.open_dataset(self.filefmt.format(runstart_init))[self.variables]\
            .astype('float16')
        data = data.rename({'time':'leadtime'})
        data = data.assign_coords(leadtime = range(len(data.leadtime)))
        
        for run in range(runstart_init+1,runstart_final+1):
            
            filepath = self.filefmt.format(run)
            nextdata = xr.open_dataset(filepath)[self.variables].\
                astype('float16')
            nextdata = nextdata.rename({'time':'leadtime'})
            nextdata = nextdata.assign_coords(leadtime = range(len(nextdata.leadtime)))
            data = xr.concat([data,nextdata],dim = 'runstart')
        
        data = data.assign_coords(runstart = range(runstart_init,runstart_final+1))
        self.fields = data
        self.dsattrs = {variable:self.fields[variable].attrs\
                        for variable in self.variables}
        self.runstart = runstart_init
        
        self.state = self.fields.sel(runstart = self.runstart)
        self.shape = np.shape(self.state[self.variables[0]])
        
        return self.state
    
    def set_dim_p(self):
        
        """Calculates dimension of state vector"""
        
        self.dim_p = np.prod(self.shape)*self.nvars
    
    def step(self, step,steps_forward):
        
        """
        Step the model forward in time from given step.  
        
        Parameters
        ----------
        step : int
            current time step
        
        Returns
        -------
        nextfield : xarray DataSet
            DataSet containing the model fields at the next timestep
        """
        
        self.runstart = step + steps_forward
        self.state = self.fields.isel(runstart = self.runstart)
        
        return self.state
        
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
        
        return state_p
    
    def nn_interpolator(self, coords,variable):
        """
        Returns indices for nearest neighbour interpolation to given
        coordinates, along with, weights for each observation.

        Parameters
        ----------
        coords : ndarray of float
            n*3 array containing leadtime, lat, lon coords for each point
        variable : str
            variable to find indices for

        Returns
        -------
        indices : ndarray of int
            n*1 array containing index in state vector of each point
        weights : ndarray of float
            n*1 array containing weights to be given to each observation

        """
        
        # convert lat and lon coords to indices in field
        for i in range(np.shape(coords)[0]):
            coords[i,1] = np.argmin(np.abs(self.fields.latitude.data - coords[i,1]))
            coords[i,2] = np.argmin(np.abs(self.fields.longitude.data - coords[i,2]))
        # convert to coordinates in state vector
        indices = np.ravel_multi_index(coords.astype('int').T, dims=self.shape,order = 'F')
        indices = np.reshape(indices, (-1,1))
        weights = np.ones_like(indices, dtype=float)
        weights = np.where(np.logical_and(indices>=0, indices<self.dim_p),
                           weights, 0.0)
        
        try:
            varindex = self.variables.index(variable)
        except:
            raise Exception('Invalid variable selected. Valid variables are ' + \
                            self.variables)
        
        indices += varindex*self.dim_p//self.nvars
        
        return indices, weights
    
    def write_output(self,savedir = ''):
        
        """
        Writes current state to netcdf file
        
        Parameters
        ----------
        savedir : str
            Directory to save output in
        """
        
        filename = savedir + 'step{:}member{:}.nc'.format(self.time,self.member)
        self.state.astype('f8').to_netcdf(filename)
        print('Saved current state at '+ filename)
        
        