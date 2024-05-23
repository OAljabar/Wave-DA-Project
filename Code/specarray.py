# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:22:07 2024

@author: owena
"""

import numpy as np
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import copy

class specarray:
    
    def __init__(self,filepath):
        
        '''
        Create an xarray dataset containing the spectral data from a given
        filepath. Also extracts the location of all gridpoints
        '''
        
        datalist = open(filepath).read().split('GOM')

        # calculate k and theta arrays from file header
        coordslist = datalist[0].split()
        k = np.array(coordslist[10:39],dtype = 'float')
        theta = np.array(coordslist[39:63],dtype = 'float')

        # get indices of gridpoints and number of gridpoints and timesteps
        indices = np.array([datalist[i][:2] for i in range(1,len(datalist))],\
                           dtype = 'int')
        npts = max(indices)
        nt = int(len(datalist)/npts)
        
        # get timestamps
        fmt = '%Y%m%d %H%M%S'
        times = [dt.datetime.strptime(datalist[i*npts][-17:-2],fmt) for i in range(nt)]

        # get spectral data from file and store in array
        data = np.zeros([nt,npts,len(k),len(theta)])
        specstart = 9
        specend = 9 + len(k)*len(theta)

        for i in range(nt*npts):
            F = np.array(datalist[i+1].split()[specstart:specend],dtype = 'float')
            F = np.reshape(F,[len(k),len(theta)],order = 'F')
            pt = int(datalist[i+1][:2]) - 1
            t = i//npts
            data[t,pt] = F

        # convert to DataArray type
        self.data = xr.DataArray(data,dims = ('time','gridpt','k','theta'),\
                            coords = {'time':times,'gridpt':range(1,npts+1),\
                                      'k':k,'theta':theta})
        
        # get lat-lon coordinates of all gridpoints
        self.lat = [float(datalist[i+1][10:15]) for i in range(npts)]
        self.lon = [float(datalist[i+1][16:22]) for i in range(npts)]
        
        # set flag to show whether spectrum is a function of k or sigma
        self.freqspecflag = False
    
    def freqspec(self):
        
        '''
        Convert wavenumber-direction spectrum to frequency-direction spectrum
        NOTE: currently assuming all gridpoints are in deep water - probably
        invalid, need to get depth data from netcdf files.
        '''
        
        g = 9.81 # gravitational acceleration in ms^-2
        
        # calculate group speed as a function of wavenumber, assuming deep water
        c_g = np.sqrt(g/self.data.k)/2
        
        # convert wavenumber spectrum to frequency spectrum using Jacobian
        spec = copy.copy(self)
        spec.data = spec.data/c_g
        
        # calculate frequency as a function of wavenumber using deep water
        # dispersion relation
        sigma = np.sqrt(g*self.data.k.data)
        
        # change coordinates of dataarray and rename
        spec.data = spec.data.assign_coords(k = sigma).rename({'k':'sigma'})
        
        # change flag to indicate frequency spectrum
        spec.freqspecflag = True
        
        return spec
    
    def inttheta(self):
        
        '''
        Integrate the spectrum with respect to direction
        '''
        spec = copy.copy(self)
        spec.data = self.data.mean(dim = 'theta')*2*np.pi
        return spec
    
    def intk(self):
        
        '''
        Integrate the spectrum with respect to wavenumber
        '''
        
        # weights for integrating with respect to wavenumber
        k = self.data.k.data
        dk = np.zeros_like(k)
        dk[0] = k[1] - k[0]
        dk[-1] = k[-1] - k[-2]
        dk[1:-1] = (k[2:] - k[:-2])/2
        weights = xr.DataArray(dk,dims = ('k'),coords = {'k':k})
        
        spec = copy.copy(self)
        spec.data = (self.data*weights).sum(dim = 'k')
        return spec
    
    def intsigma(self):
        
        # weights for integrating with respect to wavenumber
        sigma = self.data.sigma.data
        dsigma = np.zeros_like(sigma)
        dsigma[0] = sigma[1] - sigma[0]
        dsigma[-1] = sigma[-1] - sigma[-2]
        dsigma[1:-1] = (sigma[2:] - sigma[:-2])/2
        weights = xr.DataArray(dsigma,dims = ('sigma'),coords = {'sigma':sigma})
        
        spec = copy.copy(self)
        spec.data = (self.data*weights).sum(dim = 'sigma')
        return spec
    
    def specplots(self,gridpt,tmin = 0,tmax = None):
        
        '''
        Produce plots of unidirectional wavenumber spectrum and directional
        spread for a given gridpoint over a given time window. tmax is set to
        the index of the final timestep by default.
        '''
        
        if tmax is None:
            tmax = len(self.data.time)
        
        # calculate the unidirectional spectrum and directional spread
        specud = self.inttheta().data
        if self.freqspecflag:
            specdir = self.intsigma().data
        else:
            specdir = self.intk().data
        
        for i in range(tmin,tmax):
            
            fig,[uax,dax] = plt.subplots(1,2,figsize = (12,4))
            specud.isel(gridpt = gridpt-1,time = i).plot(ax = uax)
            specdir.isel(gridpt = gridpt-1,time = i).plot(ax = dax)
    
    def plotgridpts(self, buffer = 2):
        
        mapfig = plt.figure(figsize = (10,7))
        ax = mapfig.add_subplot(projection = ccrs.PlateCarree())
        ax.coastlines()
        ax.plot(self.lon,self.lat,'x')
        ax.set_xlim([min(self.lon)-buffer,max(self.lon)+buffer])
        ax.set_ylim([min(self.lat)-buffer,max(self.lat)+buffer])