# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:15:54 2024

@author: owena
"""

import numpy as np
import xarray as xr
import datetime as dt

def makeall(members,days):
    
    '''
    Creates netcdf files of all spectra from given ensemble members and
    given run start dates
    '''
    
    filefmt = '..//..//data//{:02d}//spectral//ww3.swan.gom.202203{:02d}12'
    
    for member in members:
        for day in days:
            makeds(filefmt.format(member,day),savenc = True)

def makeds(filepath,savenc = False):
    
    '''
    Reads a given WAVEWATCH III output file into an xarray dataset. Optionally
    saves as netcdf file
    
    Parameters:
    filepath : str
        path to output file
    savenc : bool
        if true, save dataset to netcdf file
    '''
    
    datalist = open(filepath).read().split('GOM')

    # calculate k and theta arrays from file header
    coordslist = datalist[0].split()
    f = np.array(coordslist[10:39],dtype = 'float')
    theta = np.array(coordslist[39:63],dtype = 'float')

    # get indices of gridpoints and number of gridpoints and timesteps
    indices = np.array([datalist[i][:2] for i in range(1,len(datalist))],\
                       dtype = 'int')
    npts = max(indices)
    nt = int(len(datalist)/npts)
    
    # get timestamps
    fmt = '%Y%m%d %H%M%S'
    times = [dt.datetime.strptime(datalist[i*npts][-17:-2],fmt) for i in range(nt)]

    # initialise arrays for spectral data as well as wind and current
    data = np.zeros([nt,npts,len(f),len(theta)])
    specstart = 9
    specend = 9 + len(f)*len(theta)
    
    wspd = np.zeros([nt,npts])
    wdir = np.zeros([nt,npts])
    cspd = np.zeros([nt,npts])
    cdir = np.zeros([nt,npts])

    # read spectra, wind and current data from file into arrays
    for i in range(nt*npts):
        F = np.array(datalist[i+1].split()[specstart:specend],dtype = 'float')
        F = np.reshape(F,[len(f),len(theta)],order = 'F')
        pt = int(datalist[i+1][:2]) - 1
        t = i//npts
        data[t,pt] = F
        wspd[t,pt] = float(datalist[i+1][35:39])
        wdir[t,pt] = float(datalist[i+1][40:45])
        cspd[t,pt] = float(datalist[i+1][48:52])
        cdir[t,pt] = float(datalist[i+1][53:58])
    
    ### convert numpy arrays to xarray DataArrays ###
    # spectra
    spec = xr.DataArray(data,dims = ('time','gridpt','f','theta'),\
                        coords = {'time':times,'gridpt':range(1,npts+1),\
                                  'f':f,'theta':theta})
    
    # wind and current
    cspd = xr.DataArray(cspd,dims = ('time','gridpt'),\
                       coords = {'time':times,'gridpt':range(1,npts+1)})
    cdir = xr.DataArray(cdir,dims = ('time','gridpt'),\
                       coords = {'time':times,'gridpt':range(1,npts+1)})
    wspd = xr.DataArray(wspd,dims = ('time','gridpt'),\
                       coords = {'time':times,'gridpt':range(1,npts+1)})
    wdir = xr.DataArray(wdir,dims = ('time','gridpt'),\
                       coords = {'time':times,'gridpt':range(1,npts+1)})
    
    # get latitude and longitude for each gridpoint
    lat = xr.DataArray([float(datalist[i+1][10:15]) for i in range(npts)],\
                       dims = ('gridpt'),coords = {'gridpt':range(1,npts+1)})
    lon = xr.DataArray([float(datalist[i+1][16:22]) for i in range(npts)],\
                       dims = ('gridpt'),coords = {'gridpt':range(1,npts+1)})
    
    # combine all variables into one dataset
    ds = xr.Dataset({'spec':spec,'lat':lat,'lon':lon,\
                     'cdir':cdir,'cspd':cspd,'wdir':wdir,'wspd':wspd})
    
    if savenc:
        savepath = filepath + '.specarray.nc'
        print('Saved spectrum at ',savepath)
        ds.to_netcdf(savepath)
    
    return ds