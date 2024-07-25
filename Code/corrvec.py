# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:01:01 2024

@author: owena
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import datetime as dt
import matplotlib.colors as colors

def corrvec(data,lat,lon,time = None,plot = False,\
            split = 'rows'):
    
    '''
    Calculates the vector of correlation coefficients with a specified point.
    Returns correlation in a 1d vector, along with vectors containing lat and
    lon for each corresponding point. Order specifies whether to split spatial
    data into rows or columns when flattening.
    '''
    
    # index ij to split into rows, xy to split into columns
    # lats,lons = np.meshgrid(data.latitude,data.longitude,indexing = 'ij')
    # lats = lats.flatten()
    # lons = lons.flatten()
    
    if not time is None:
        data = data.sel(time = time)
        if type(time) is slice:
            title = 'Times between ' + time.start.strftime('%Y-%m-%dT%H:%M') + \
                ' and ' + time.stop.strftime('%Y-%m-%dT%H:%M')
        else:
            title = 'Time = ' + time.strftime('%Y-%m-%dT%H:%M')
    else:
        title = 'All timesteps'
    
    lats = data.latitude.data
    lons = data.longitude.data
    
    h1 = data.sel(latitude = lat,longitude = lon).data.flatten()
    
    corrs = np.zeros([len(lats),len(lons)])
    
    for i in range(len(lats)):
        for j in range(len(lons)):
            
            h2 = data.sel(latitude = lats[i],longitude = lons[j]).data.flatten()
        
            if np.isnan(h2[0]):
                corrs[i,j] = np.nan
            else:
                corrs[i,j] = np.corrcoef(h1,h2,rowvar = False)[0,1]
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(projection = ccrs.PlateCarree())
        ax.coastlines()
        CS = ax.pcolormesh(lons,lats,corrs,\
                         norm = colors.CenteredNorm(),cmap = 'RdBu_r')
        cbar = plt.colorbar(CS)
        cbar.set_label('Correlation with marked point')
        ax.plot(lon,lat,'wx')
        ax.set_title(title)
        
    ### MAKE SURE THIS IS RIGHT ###
    orders = {'rows':'C','columns':'F'}
    indexings = {'rows':'ij','columns':'xy'}
    
    # index ij to split into rows, xy to split into columns
    lats,lons = np.meshgrid(data.latitude,data.longitude,indexing = indexings[split])
    lats = lats.flatten()
    lons = lons.flatten()
    
    corrs = np.reshape(corrs,len(lats),order = orders[split])
    
    return corrs,lats,lons