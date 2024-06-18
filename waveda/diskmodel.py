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
import os
import xarray as xr
from oop.models import Model

FILEDIR = os.environ.get('FILEDIR')
FILEFMT_COARSE = FILEDIR + '/{:02d}/netcdf/ww3.sgom.202203{:02}12.nc'
FILEFMT_FINE = FILEDIR + '/{:02d}/netcdf/ww3.gom.202203{:02}12.nc'

class DiskModel(Model):
    
    def __init__(self,run,member,starttime = 0,filefmt=FILEFMT_FINE):
        
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
        self.dt = 1.
        self.filepath = filefmt.format(member,run)
        self.run = run
        self.member = member
        self.step_init= int(starttime)
        self.time_init = float(starttime)
        
    def init_fields(self, process_control):
        
        """
        Initialises model field at beginning of model run
        """
        self.pe = process_control 
        
        self.step = self.step_init
        self.time_init = self.step_init * self.dt
        
        self.fields = xr.open_dataset(self.filepath)[['hs','t01','t02']].astype('float16')
        self.dsattrs = {variable:self.fields[variable].attrs for variable in list(self.fields.variables)}
        
        self.state = self.fields.isel(time = self.step)
      
    @property   
    def dim_state(self):
        nlat = len(self.fields.latitude)
        nlon = len(self.fields.longitude)
        return 3*nlon*nlat
        
    def step_forward(self, step, steps_forward):
        
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
        
        self.step = step + steps_forward 
        self.state = self.fields.isel(time = self.step)
        
        return self.state
        
    def collect_state_pdaf(self, dim_p, state_p):
        hs = np.reshape(self.state['hs'].data,int(dim_p/3),order = 'F')
        tm01 = np.reshape(self.state['t01'].data,int(dim_p/3),order = 'F')
        tm02 = np.reshape(self.state['t02'].data,int(dim_p/3),order = 'F')
        state_p = np.concatenate((hs,tm01,tm02))
        
        return state_p
    
    def distribute_state_pdaf(self, dim_p, state_p):
        
        nlat = len(self.fields.latitude)
        nlon = len(self.fields.longitude)
        
        hs = np.reshape(state_p[:int(dim_p/3)],[nlat,nlon],order = 'F')
        tm01 = np.reshape(state_p[int(dim_p/3):int(2*dim_p/3)],[nlat,nlon],order = 'F')
        tm02 = np.reshape(state_p[int(2*dim_p/3):],[nlat,nlon],order = 'F')
        
        #TODO: This doesn't work. The information in state_p contains the analysis. It needs to be saved into this object and/or
        #in case of this disk model written to output files. 
        data = xr.Dataset(data_vars = {'hs':(['latitude','longitude'],hs,self.dsattrs['hs']),\
                                          't01':(['latitude','longitude'],tm01,self.dsattrs['t01']),\
                                          't02':(['latitude','longitude'],tm02,self.dsattrs['t02'])},\
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
        
        filename = os.path.join(savedir,'step{:}member{:}.nc'.format(self.step,self.member))
        self.state.astype('f8').to_netcdf(filename)
        print('Saved current state at '+ filename)
        
    def nn_interpolate(self, field, coord):
        #TODO: return index, weight with index a (np.size(coord,0),1) array with indices linking geospatial coordinates to index in state_p 
        #and weight the array np.ones((np.size(coord,0),1))