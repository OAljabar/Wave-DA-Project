# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:37:44 2024

@author: owena
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from specarray import specarray
import datetime as dt

nmembers = 15 # number of ensemble members
pathfmt = '..//..//data//{:02d}//spectral//ww3.swan.gom.202203{:02d}12'

time = dt.datetime(2022,3,11,22)
leadtimes = []

# load spectrum from first ensemble member
ds = specarray(pathfmt.format(1,6))
ds.data = ds.data.sel(time = time)
for i in range(2,nmembers+1):
    # get data for next ds member
    nextdata = specarray(pathfmt.format(i,6)).data.sel(time = time)
    # concatenate with existing ensemble members
    ds.data = xr.concat([ds.data,nextdata],dim = 'member')
leadtimes.append(time - dt.datetime(2022,3,6,12))

for day in range(7,12):
    
    # load spectrum from first ensemble member
    ensemble = specarray(pathfmt.format(1,day))
    ensemble.data = ensemble.data.sel(time = time)
    for i in range(2,nmembers+1):
        # get data for next ensemble member
        nextdata = specarray(pathfmt.format(i,day)).data.sel(time = time)
        # concatenate with existing ensemble members
        ensemble.data = xr.concat([ensemble.data,nextdata],dim = 'member')
    leadtimes.append(time - dt.datetime(2022,3,day,12))
    ds.data = xr.concat([ds.data,ensemble.data],dim = 'leadtime')

ds.data = ds.data.assign_coords(member = range(1,nmembers+1),leadtime = leadtimes)

# Plot variance as a function of t and f, averaged over gridpt and direction
V = ds.data.var(dim = 'member')
V = V.mean(dim = ['gridpt','theta'])
plt.figure()
plt.contourf(V.leadtime,V.f,V.data.transpose())
plt.xlabel('Lead time (days)')
plt.ylabel('frequency (Hz)')
plt.title('Ensemble Variance')

# Plot energy plume
E = ds.intf().inttheta()
fig,ax = plt.subplots(figsize = (10,7))
E.data.sel(gridpt = 26).plot.line(ax = ax,x = 'leadtime',add_legend = False,color = 'k')
