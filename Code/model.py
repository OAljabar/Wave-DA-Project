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
    
    def __init__(self,run,member,starttime = 0,\
                 filefmt = '/storage/silver/metstudent/msc/users_2024/mg838076/data/{:02d}/netcdf/ww3.sgom.202203{:02}12.nc'):
        
        """
        Creates model object storing fields to be corrected from a given model
        run
        
        Parameters
        ----------
        run : int
            Start date of model run to be used
        member : int
            Ensemble member to be used
        starttime : int, optional
            Timestep to begin the assimilation from. Default is 0
        filefmt : str, optional
            Format string pointing to the location in storage of the files
            containing model fields
        
        Attributes
        ----------
        filepath : str
            String pointing to the location of the specific file to be used
        un : int
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
        
        self.filepath = filefmt.format(member,run)
        self.run = run
        self.member = member
        self.starttime = starttime
        self.time = starttime
        self.fields = xr.open_dataset(self.filepath)[['hs','t01','t02']].\
            astype('float16')
        
    def init_fields(self):
        
        """
        Initialises model field at beginning of model run
        """
        
        self.state = self.fields.isel(time = self.starttime)
        
        return self.state
        
    def step(self, step):
        
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
        
        self.time = step + 1
        self.state = self.fields.isel(time = self.time)
        
        return self.state
        
    def collect_state_pdaf(self, dim_p, state_p):
        
        """Outputs local model fields as array to be collected by PDAF.

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
        
        hs = np.reshape(self.state['hs'].data,dim_p,order = 'F')
        tm01 = np.reshape(self.state['t01'].data,dim_p,order = 'F')
        tm02 = np.reshape(self.state['t02'].data,dim_p,order = 'F')
        state_p = np.concatenate((hs,tm01,tm02))
        
        return state_p
    
    def distribute_state_pdaf(self, dim_p, state_p):
        
        nlat = len(self.fields.latitude)
        nlon = len(self.fields.longitude)
        
        hs = np.reshape(state_p[:int(dim_p/3)],[nlat,nlon],order = 'F')
        tm01 = np.reshape(state_p[int(dim_p/3):int(2*dim_p/3)],[nlat,nlon],order = 'F')
        tm02 = np.reshape(state_p[int(2*dim_p/3):],[nlat,nlon],order = 'F')
        
        state_p = xr.Dataset(data_vars = {'hs':(['latitude','longitude'],hs),\
                                          't01':(['latitude','longitude'],tm01),\
                                          't02':(['latitude','longitude'],tm02)},\
                             coords = {'latitude':self.fields.latitude,\
                                       'longitude':self.fields.longitude})
        
        return state_p
    
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
        
        