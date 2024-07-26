# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:37:44 2024

@author: owena
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import datetime as dt

class specenstime:
    
    def __init__(self,time,nmembers = 10):
        
        '''
        Creates an ensemble of wave spectra for the given time, using all
        available model runs.
        '''
        
        filefmt = '..//..//data//{:02d}//spectral//ww3.swan.gom.202203{:02d}12.specarray.nc'
        
        # initialise arrays for lead times and ensemble data
        leadtimes = []
        ds = xr.DataArray()
        
        # determine earliest and latest model run which can be included in ensemble
        firstday = time.day - 7
        lastday = time.day
        if time.hour >= 12:
            firstday += 1
            lastday += 1
        firstday = max(6,firstday)
        lastday = min(13,lastday)
        
        # check time given is inside forecast window
        if time > dt.datetime(2022,3,19,12) or time < dt.datetime(2022,3,6,12):
            raise ValueError('Time given must be between 06/03/2022 12:00 and \
                             19/03/2022 12:00')
        # check number of ensemble members given is valid
        if nmembers < 1 or nmembers > 30:
            raise ValueError('Number of ensemble members must be between 1 \
                             and 30')
        
        # loop over model runs
        for day in range(firstday,lastday):
            # get data for the correct number of ensemble members
            ensemble = xr.open_dataset(filefmt.format(1,day))['spec'].sel(time = time)
            for member in range(2,nmembers+1):
                data = xr.open_dataset(filefmt.format(member,day))['spec'].sel(time = time)
                ensemble = xr.concat([ensemble,data],dim = 'member')
            ds = xr.concat([ds,ensemble],dim = 'leadtime')
            # add lead time for this model run to list
            leadtimes.append(time - dt.datetime(2022,3,day,12))
        
        #remove empty part of array
        ds = ds.isel(leadtime = slice(1,len(ds.leadtime)))
        
        # assign correct coordinates for leadtime and member
        ds = ds.assign_coords(member = range(1,nmembers+1),leadtime = leadtimes)
        ds = ds.squeeze()
        
        self.data = ds
        
        self.pos = xr.open_dataset(filefmt.format(member,day))[['lat','lon']]
    
    def ensemblevar(self,gridpt = None):
        
        '''
        Calculates the ensemble variance of the unidirectional spectrum at the
        given gridpoint, or averaged over all points if no gridpoint is given.
        '''
        
        # calculate unidirectional spectrum
        specud = self.data.mean(dim = 'theta')*2*np.pi
        if gridpt is None:
            specud = specud.mean(dim = 'gridpt')
        else:
            specud = specud.sel(gridpt = gridpt)
        
        # return ensemble variance
        return specud.var(dim = 'member')
    
    def varplot(self,gridpt = None,freq = None):
        
        '''
        Plots the variance as a function of leadtime and frequency for a given
        gridpoint and frequency range. If no gridpoint is given, averages over
        all gridpoints. If no frequency range is given, plots entire range of
        available frequencies.
        '''
        
        # calculate variance and select frequencies if required
        V = self.ensemblevar(gridpt)
        if not freq is None:
            V = V.sel(f = freq)
        
        # create title for plot
        if gridpt is None:
            title = 'Ensemble variance (all gridpoints)'
        else:
            pos = self.pos.sel(gridpt = gridpt)
            title = 'Ensemble variance at ({lat},{lon})'.format(lat = pos['lat'].data,lon = pos['lon'].data)
        
        # add frequency to title if a single frequency is given
        if len(V.dims) == 1:
            title += ', period = {:} s'.format(1/freq)
        
        period = 1/V.f
        lts = V.leadtime.data.astype('float64')/86400000000000
        
        # plot variance
        fig,ax = plt.subplots()
        # line plot if single frequency selected, contour plot otherwise
        if len(V.dims) == 1:
            ax.plot(lts,V.data)
            ax.set_ylabel('Variance ($m^4/Hz^2$)')
            ax.set_ylim(bottom = 0)
        else:
            CS = ax.contourf(lts,period,V.data.transpose())
            ax.set_ylabel('Period (s)')
            fig.colorbar(CS,label = 'Variance ($m^4/Hz^2$)')
        ax.set_xlabel('Lead time (days)')
        ax.set_title(title)
'''
# Plot energy plume
E = ds.intf().inttheta()
fig,ax = plt.subplots(figsize = (10,7))
E.data.sel(gridpt = 26).plot.line(ax = ax,x = 'leadtime',add_legend = False,color = 'k')
'''