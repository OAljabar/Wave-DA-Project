# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:40:13 2024

@author: owena
"""

import xarray as xr
import scipy.stats as stats

class twinobs:
    
    def __init__(self,member = 0,run = 6,finedomain = False,variable = 'hs'):
        
        '''
        Generates a dataset to represent the true state of the ocean in a twin
        experiment from which artifical observations can be calculated
        '''
        
        if finedomain:
            filepath = '..//..//data//{:02d}//netcdf//ww3.sgom.202203{:02d}12.nc'\
                .format(member,run)
        else:
            filepath = '..//..//data//{:02d}//netcdf//ww3.gom.202203{:02d}12.nc'\
                .format(member,run)
        
        self.data = xr.open_dataset(filepath)[variable]
    
    def obs(self,pos,times,errscale = 0.065):
        
        '''
        Generates an artificial observation at a given location and times, with
        a given standard deviation of errors
        '''
        
        # interpolate to find value
        value = self.data.interp(latitude = pos[0],longitude = pos[1],\
                                 time = times)
        
        # generate noise from normal distribution
        noise = stats.norm.rvs(scale = errscale*value,size = len(times))
        
        return value + xr.DataArray(noise,dims = ['time'],coords = {'time':times})