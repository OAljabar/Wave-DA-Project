# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:27:29 2024

@author: owena
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from ensembleheight import heightensrun
import matplotlib as mpl
from datetime import datetime

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
    ds = xr.open_dataset(filepath)[['uwnd','vwnd','dir','dp','fp','dpt']]
    
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
    
    g = 9.81
    ds['cgp'] = g/(4*np.pi*ds['fp'])
    
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
    ds.plot.quiver(x = 'longitude',y = 'latitude',u = 'udp',v = 'vdp',hue = 'cgp',ax = ax)
    ax.coastlines()
    ax.set_title('Wave peak direction, ' + timestr)
    
    plt.figure()
    ds['dpt'].plot()

def wcorrmaps(run,time = None,member = None,finedomain = False,thinfactor = 5,\
              var1 = 'hs',var2 = 'hs',ens = None,wdata = None):
    
    '''
    Produces maps of the wind and mean wave direction at a given time
    '''
    if wdata is None:
        # open file as dataset
        ds = heightensrun(run,variables = ['uwnd','vwnd']).data.mean(dim = 'member')
    else:
        ds = wdata
    
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
    
    # get wave data if not supplied
    if ens is None:
        ens = heightensrun(run,variables = [var1,var2])
    
    # calculate magnitude and normalised wind values
    #ds['magnitude'] = np.sqrt(ds['uwnd']**2 + ds['vwnd']**2)
    #ds['unorm'] = ds['uwnd']/ds['magnitude']
    #ds['vnorm'] = ds['vwnd']/ds['magnitude']
    
    # plot wind arrows
    fig = plt.figure(figsize = (8,7))
    ax = fig.add_subplot(projection = ccrs.PlateCarree())
    ens.corrvec(18.4,-93.8,time,True,ax,var1 = var1,var2 = var2);
    ds.plot.quiver(x = 'longitude',y = 'latitude',u = 'uwnd',v = 'vwnd',\
                   ax = ax)
    ax.coastlines()
    ax.set_title('Wind, ' + timestr)
    
def hdirmaps(start,end,step = 6,run = None,member = None,data = None,\
             thinfactor = 10,plotsettings = {}):
    
    if data is None:
        data = heightensrun(run,variables = ['hs','dir']).data
    
    thindata = data.thin({'latitude':thinfactor,'longitude':thinfactor})
    
    if 'member' in list(data.dims):
        data = data.mean(dim = 'member')
    
    maxheight = np.nanmax(data['hs'].data)
    
    with mpl.rc_context(plotsettings):
    
        for i in range(start,end,step):
            thinstep = thindata.isel(time = i)
            step = data.isel(time = i)
            thinstep['udir'] = -np.sin(np.deg2rad(thinstep['dir']))
            thinstep['vdir'] = -np.cos(np.deg2rad(thinstep['dir']))
            #print(step)
            
            fig = plt.figure(figsize = (9,7))
            ax = fig.add_subplot(projection = ccrs.PlateCarree())
            ax.coastlines()
            step['hs'].plot.pcolormesh(ax = ax,cmap = 'Blues',vmin = 0,vmax = maxheight,\
                                       cbar_kwargs = {'label':'Significant wave height (m)'})
            ax.quiver(x = thinstep.longitude,y = thinstep.latitude,\
                      u = thinstep['udir'],v = thinstep['vdir'])
            ax.set_title(np.datetime_as_string(data.time.data[i],'m'))
            fig.savefig('../../animation/frames/{:03d}'.format(i))
            plt.clf()