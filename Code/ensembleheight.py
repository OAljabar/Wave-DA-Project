# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:16:32 2024

@author: owena
"""

import xarray as xr
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import skgstat as skg

class heightensrun:
    
    def __init__(self,run,nmembers = 30):
        
        '''
        Creates an ensemble of significant wave height from a given model
        run.
        '''
        
        # check model run given is valid
        if run < 6 or run > 12:
            raise ValueError('Model runs only available between 06/03/2022\
                             and 12/03/2022')
        # check number of ensemble members given is valid
        if nmembers < 1 or nmembers > 30:
            raise ValueError('Number of ensemble members must be between 1 \
                             and 30')
        
        # filepath format for netcdf files
        filefmt = '..//..//data//{{:02d}}//netcdf//ww3.sgom.202203{:02d}12.nc'\
            .format(run)
        
        # get data for first ensemble member
        ens = xr.open_dataset(filefmt.format(1))['hs']
        
        for member in range(2,nmembers+1):
            # get data from next member and add to array
            nextmember = xr.open_dataset(filefmt.format(member))['hs']
            ens = xr.concat([ens,nextmember],dim = 'member')
        
        # assign coordinates to ensemble members
        ens = ens.assign_coords(member = range(1,nmembers+1))
        
        self.data = ens
        
    def ensemblevar(self):
        
        '''
        Calculates the variance of the ensemble
        '''
        
        return self.data.var(dim = 'member')
    
    def varplot(self,lat = None,lon = None):
        
        '''
        Creates a plot of ensemble mean and variance as a function of time
        '''
        
        M = self.data.mean(dim = 'member')
        V = self.ensemblevar()
        if lat is None:
            M = M.mean(dim = 'latitude')
            V = V.mean(dim = 'latitude')
        else:
            M = M.sel(latitude = lat)
            V = V.sel(latitude = lat)
        if lon is None:
            M = M.mean(dim = 'longitude')
            V = V.mean(dim = 'longitude')
        else:
            M = M.sel(longitude = lon)
            V = V.sel(longitude = lon)
        
        fig,axs = plt.subplots(2,1,figsize = (6,10))
        M.plot(ax = axs[0])
        V.plot(ax = axs[1])
        axs[0].set_title('Ensemble Mean')
        axs[1].set_title('Ensemble Variance')
        fig.tight_layout()
        
    
    def tvariogram(self,lat = None,lon = None):
        
        '''
        Computes a temporal variogram from ensemble data. Selects data from
        a specific latitude and longitude if given, otherwise averages over
        spatial points
        '''
        
        # select/average over spatial points
        data = self.data
        if lat is None:
            data = data.mean(dim = 'latitude')
        else:
            data = data.sel(latitude = lat)
        if lon is None:
            data = data.mean(dim = 'longitude')
        else:
            data = data.sel(longitude = lon)
        
        # flatten data and create time array in days to pass to variogram class
        nmembers = len(self.data.member)
        t = np.tile(data.time.astype('float')/86400000000000,nmembers)
        data = data.data.flatten()
        
        # create and plot variogram
        V = skg.Variogram(t,data)
        fig,ax = plt.subplots()
        V.plot(axes = ax,hist = False)
        ax.set_xlabel('Lag (days)')
        
        return V
    
    def xvariogram(self,time = None):
        
        '''
        ###Currently having memory issues trying to run this###
        Creates a spatial variogram from ensemble data. Selects data from 
        a specific timestep if given, otherwise averages over all times
        TO FIX: distance is currently calculated using euclidean metric on 
        lat-lon positions
        '''
        
        raise Exception('This method currently runs into memory problems \
                        when trying to create the variogram. Maybe some\
                            localisation is needed?')
        
        # select/average over times
        data = self.data
        if time is None:
            data = data.mean(dim = 'time')
        else:
            data = data.sel(time = time)
        
        # create lat-lon array to pass to variogram class
        lat,lon = np.meshgrid(data.latitude,data.longitude)
        x = np.stack((lat.flatten(),lon.flatten()),axis = 1)
        nmembers = len(self.data.member)
        x = np.tile(x,(nmembers,1))
        
        data = data.data.flatten()
        
        # create and plot variogram
        V = skg.Variogram(x,data)
        fig,ax = plt.subplots()
        V.plot(axes = ax,hist = False)
        ax.set_xlabel('Distance')
        
        return V
    
    def enshist(self,lat = 21.15,lon= -95,time = dt.datetime(2022,3,12,22)):
        
        '''
        Generates histogram of ensemble members for a specific position
        and time
        '''
        
        pt = self.data.sel(latitude = lat,longitude = lon,time = time)
        #fig,ax = plt.subplots()
        pt.plot.hist()

class heightenstime:
    
    def __init__(self,time,nmembers = 30):
    
        '''
        Creates an ensemble of significant wave height for the given time.
        '''
        
        # filepath format for netcdf files
        filefmt = '..//..//data//{:02d}//netcdf//ww3.gom.202203{:02d}12.nc'
        
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
            ensemble = xr.open_dataset(filefmt.format(1,day))['hs'].sel(time = time)
            for member in range(2,nmembers+1):
                data = xr.open_dataset(filefmt.format(member,day))['hs'].sel(time = time)
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
    
    def ensemblevar(self):
        
        '''
        Calculates the variance of the ensemble
        '''
        
        return self.data.var(dim = 'member')
    
    def varplot(self,lat = None,lon = None):
        
        '''
        Creates a plot of ensemble variance as a function of lead time
        '''
        
        V = self.ensemblevar()
        if lat is None:
            V = V.mean(dim = 'latitude')
        else:
            V = V.sel(latitude = lat)
        if lon is None:
            V = V.mean(dim = 'longitude')
        else:
            V = V.sel(longitude = lon)
        
        # convert lead times to days
        lts = [float(lt)/86400000000000 for lt in V.leadtime]
        
        # plot variance as a function of lead time
        fig,ax = plt.subplots()
        ax.plot(lts,V.data)
        ax.set_xlabel('Lead Time (days)')
        ax.set_ylabel('Variance (m^2)')
        ax.set_title('Time: ' + np.datetime_as_string(V.time[()],unit = 'm'))
    
    def varmaps(self):
        
        V = self.ensemblevar()
        for lt in V.leadtime:
            plt.figure()
            V.sel(leadtime = lt).plot()