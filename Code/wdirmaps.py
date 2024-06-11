# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:27:29 2024

@author: owena
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

def wdirmaps(run,time = None,member = 0,finedomain = False,thinfactor = 5):
    
    '''
    Produces maps of the wind and mean wave direction at a given time
    '''
    
    # get path of netcdf file
    if finedomain:
        filefmt = '..//..//data//{:02d}//netcdf//ww3.sgom.202203{:02d}12.nc'
    else:
        filefmt = '..//..//data//{:02d}//netcdf//ww3.gom.202203{:02d}12.nc'
    
    filepath = filefmt.format(member,run)
    
    # open file as dataset
    ds = xr.open_dataset(filepath)[['uwnd','vwnd','dir','dp']]
    
    # select specified time and thin data for plotting
    if time is None:
        ds = ds.mean(dim = 'time')
        timestr = 'temporal mean'
    elif type(time) is slice:
        ds = ds.sel(time = time).mean(dim = 'time')
        timestr = 'temporal mean from ' + time.start.strftime('%Y-%m-%dT%H:%M')\
            + ' to ' + time.stop.strftime('%Y-%m-%dT%H:%M')
    else:
        ds = ds.sel(time = time)
        timestr = 'time = ' + time.strftime('%Y-%m-%dT%H:%M')
    
    ds = ds.thin(thinfactor)
    
    # convert meteorological convention directions to u and v
    ds['udir'] = -np.sin(np.deg2rad(ds['dir']))
    ds['vdir'] = -np.cos(np.deg2rad(ds['dir']))
    ds['udp'] = -np.sin(np.deg2rad(ds['dp']))
    ds['vdp'] = -np.cos(np.deg2rad(ds['dp']))
    
    # make plots
    fig = plt.figure(figsize = (8,7))
    ax = fig.add_subplot(projection = ccrs.PlateCarree())
    ds.plot.quiver(x = 'longitude',y = 'latitude',u = 'uwnd',v = 'vwnd',ax = ax)
    ax.coastlines()
    ax.set_title('Wind, ' + timestr)
    
    fig = plt.figure(figsize = (8,7))
    ax = fig.add_subplot(projection = ccrs.PlateCarree())
    ds.plot.quiver(x = 'longitude',y = 'latitude',u = 'udir',v = 'vdir',ax = ax)
    ax.coastlines()
    ax.set_title('Mean wave direction, ' + timestr)
    
    fig = plt.figure(figsize = (8,7))
    ax = fig.add_subplot(projection = ccrs.PlateCarree())
    ds.plot.quiver(x = 'longitude',y = 'latitude',u = 'udp',v = 'vdp',ax = ax)
    ax.coastlines()
    ax.set_title('Wave peak direction, ' + timestr)
    