# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:05:08 2024

@author: owena
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

def curmap(run,time,finedomain = False,thinfactor = 2):
    
    '''
    Produces a map showing currents at a given time from a given model run
    '''
    
    if finedomain:
        filepath = '..//..//data//01//netcdf//ww3.sgom.202203{:02d}12.nc'
    else:
        filepath = '..//..//data//01//netcdf//ww3.gom.202203{:02d}12.nc'
    
    filepath = filepath.format(run)
    
    ds = xr.open_dataset(filepath)[['ucur','vcur']]
    
    ds = ds.sel(time = time).thin(thinfactor)
    ucur = ds['ucur']
    vcur = ds['vcur']
    
    lats = []
    lons = []
    ucurs = []
    vcurs = []
    
    for lat in ds.latitude:
        for lon in ds.longitude:
            
            if not np.isnan(ucur.sel(latitude = lat,longitude = lon)):
                lats.append(lat)
                lons.append(lon)
                ucurs.append(ucur.sel(latitude = lat,longitude = lon).data[()])
                vcurs.append(vcur.sel(latitude = lat,longitude = lon).data[()])
    
    #plt.quiver(lons,lats,ucurs,vcurs,scale = 10)
    
    return lats,lons,ucurs,vcurs