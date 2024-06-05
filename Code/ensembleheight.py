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
import scipy.stats as stats
import cartopy.crs as ccrs
import matplotlib.colors as colors

class heightensrun:
    
    def __init__(self,run = None,nmembers = 30,finedomain = False,data = None):
        
        '''
        Creates an ensemble of significant wave height from a given model
        run.
        '''
        
        if data is None:
            
            if run is None:
                raise ValueError('If no data is given, run start time and ensemble size must be specified')
        
            # check model run given is valid
            if run < 6 or run > 12:
                raise ValueError('Model runs only available between 06/03/2022\
                                 and 12/03/2022')
            # check number of ensemble members given is valid
            if nmembers < 1 or nmembers > 30:
                raise ValueError('Number of ensemble members must be between 1 \
                                 and 30')
            
            # filepath format for netcdf files
            if finedomain:
                filefmt = '..//..//data//{{:02d}}//netcdf//ww3.sgom.202203{:02d}12.nc'\
                    .format(run)
            else:
                filefmt = '..//..//data//{{:02d}}//netcdf//ww3.gom.202203{:02d}12.nc'\
                    .format(run)
            
            # get data for first ensemble member
            ens = xr.open_dataset(filefmt.format(1))[['hs','t01','t02']]
            
            for member in range(2,nmembers+1):
                # get data from next member and add to array
                nextmember = xr.open_dataset(filefmt.format(member))[['hs','t01','t02']]
                ens = xr.concat([ens,nextmember],dim = 'member')
            
            # assign coordinates to ensemble members
            ens = ens.assign_coords(member = range(1,nmembers+1))
            ens['hs^2'] = ens['hs']**2
            ens['t02^2'] = ens['t02']**2
            
            self.data = ens
        
        else:
            self.data = data
    
    def spatialmean(self,variable = None):
        
        if not variable is None:
            data = self.data[variable]
        else:
            data = self.data
        
        weights = np.cos(np.deg2rad(data.latitude))
        dataweighted = data.weighted(weights)
        mean = dataweighted.mean(dim = ('latitude','longitude'))
        
        return heightensrun(data = mean)
        
    def ensemblevar(self,variable = 'hs'):
        
        '''
        Calculates the ensemble variance for the given variable.
        '''
        
        return self.data[variable].var(dim = 'member')
    
    def varplot(self,pos = None,variable = 'hs'):
        
        '''
        Creates a plot of ensemble mean and variance as a function of time.
        Uses data from a specific point if given, otherwise averages over domain
        '''
        
        title = variable + ' ensemble mean and variance'
        
        M = self.data[variable].mean(dim = 'member')
        V = self.ensemblevar(variable)
        if pos is None:
            M = heightensrun(data = M).spatialmean().data
            V = heightensrun(data = V).spatialmean().data
            title += ' - spatial mean'
        else:
            M = M.sel(latitude = pos[0],longitude = pos[1])
            V = V.sel(latitude = pos[0],longitude = pos[1])
            title += ' at ({lat},{lon})'.format(lat = pos[0],lon = pos[1])
        
        fig,axs = plt.subplots(2,1,figsize = (6,8))
        plt.xticks(rotation=45, ha="right")
        axs[0].plot(M.time,M.data)
        axs[0].set_ylabel('Mean (m)')
        axs[1].plot(V.time,V.data)
        axs[1].set_ylabel('Variance ($m^2$)')
        axs[1].set_ylim(bottom = 0)
        fig.suptitle(title)
        fig.tight_layout()
    
    
    def varmaps(self,times,variable = 'hs'):
        
        '''
        Plots maps of ensemble mean and variance for given times
        '''
        
        M = self.data[variable].mean(dim = 'member')
        V = self.ensemblevar(variable)
        
        for time in times:
            fig,axs = plt.subplots(ncols = 2,figsize = (12,4))
            M.sel(time = time).plot(ax = axs[0])
            V.sel(time = time).plot(ax = axs[1])
            if type(time) is np.datetime64:
                tstr = 'Time = ' + np.datetime_as_string(time,unit = 'm')
            else:
                tstr = 'Time = ' + time.strftime('%Y-%m-%dT%H:%M')
            axs[0].set_title('Mean, ' + tstr)
            axs[1].set_title('Variance, ' + tstr)
    
    def tvariogram(self,lat = None,lon = None,variable = 'hs'):
        
        '''
        Computes a temporal variogram from ensemble data. Selects data from
        a specific latitude and longitude if given, otherwise averages over
        spatial points
        '''
        
        # select/average over spatial points
        data = self.data[variable]
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
        data = self.data['hs']
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
    
    def enshist(self,pos,time,xspread = 0,tspread = 0,variable = 'hs'):
        
        '''
        Generates histogram of ensemble members for a specific position
        and time
        '''
        tspread = dt.timedelta(hours = tspread)
        pt = self.data[variable].sel(latitude = slice(pos[0]-xspread,pos[0]+xspread),\
                           longitude = slice(pos[1]-xspread,pos[1]+xspread),\
                               time = slice(time-tspread,time+tspread))
        data = pt.data.flatten()
        data = (data - np.mean(data))/np.std(data)
        
        fig,ax = plt.subplots()
        binwidth = 0.5
        bins = np.arange(np.floor(min(data)),np.ceil(max(data))+0.1,binwidth)
        ax.hist(data,bins)
        
        x = np.linspace(min(bins),max(bins),100)
        
        # fit and plot gamma distribution
        [a,loc,scale] = stats.gamma.fit(data)
        ax.plot(x,stats.gamma.pdf(x,a,loc,scale)*len(data)*binwidth,'k--',\
                label = 'Gamma Distribution')
        # might need a different version to calculate fits
        #gfit = stats.goodness_of_fit(stats.gamma,data,\
                                     #fit_params = {'a':a,'loc':loc,'scale':scale})
        
        # fit and plot normal distribution
        [loc,scale] = stats.norm.fit(data)
        ax.plot(x,stats.norm.pdf(x,loc,scale)*len(data)*binwidth,'k:',\
                label = 'Normal Distribution')
        
        # fit and plot log normal distribution
        [s,loc,scale] = stats.lognorm.fit(data)
        ax.plot(x,stats.lognorm.pdf(x,s,loc,scale)*len(data)*binwidth,'k-.',\
                label = 'Log Normal Distribution')
        
        ax.set_xlim([min(bins),max(bins)])
        ax.legend()
        title = variable + ' centred on position ({:},{:}) and time '.format(pos[0],pos[1]) +\
            time.strftime('%Y-%m-%dT%H:%M')
        ax.set_title(title)
        
        return data
    
    def peakspread(self,pos = None,plot = False,variances = False):
        
        # select/average over gridpoints
        if pos is None:
            data = self.data['hs'].mean(dim = ('latitude','longitude'))
            title = 'Spatial mean wave height'
        else:
            data = self.data['hs'].sel(latitude = pos[0],longitude = pos[1])
            title = 'Wave height at ({lat},{lon})'.format(lat = pos[0],lon = pos[1])
        
        # initialise arrays to store peak times and heights
        times = np.zeros_like(data.time)[:len(data.member)]
        heights = np.zeros(len(data.member))
        
        for i,member in enumerate(data.member):
            
            mdata = data.sel(member = member).data
            maxindex = np.argmax(mdata)
            times[i] = data.time.data[maxindex]
            heights[i] = mdata[maxindex]
        
        if plot:
            fig,ax = plt.subplots()
            plt.xticks(rotation = 45,ha = 'right')
            ax.plot(times,heights,'kx')
            ax.set_xlabel('Time of peak (h)')
            ax.set_ylabel('Height of peak (m)')
            ax.set_title(title)
        
        if variances:
            # convert times to hours and calculate variances
            tvar = np.var(times.astype('float')/3.6e12)
            hvar = np.var(heights)
            
            if plot:
                text = 'Variance in height is {:.2f} $m^2$ \n Variance in time is {:.2f} $h^2$'\
                    .format(hvar,tvar)
                ax.text(max(times),max(heights),text,ha = 'right',va = 'top')
            
            return heights,times,tvar,hvar
        
        return heights,times
    
    def corrvec(self,lat,lon,time = None,plot = False,\
                split = 'rows',var1 = 'hs',var2 = 'hs'):
        
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
        
        title = 'Variables (' + var1 + ',' + var2 + '), '
        
        if not time is None:
            data1 = self.data[var1].sel(time = time)
            data2 = self.data[var2].sel(time = time)
            if type(time) is slice:
                title += 'times between ' + time.start.strftime('%Y-%m-%dT%H:%M') + \
                    ' and ' + time.stop.strftime('%Y-%m-%dT%H:%M')
            else:
                title += 'time = ' + time.strftime('%Y-%m-%dT%H:%M')
        else:
            data1 = self.data[var1]
            data2 = self.data[var2]
            title += 'all timesteps'
        
        lats = data1.latitude.data
        lons = data2.longitude.data
        
        h1 = data1.sel(latitude = lat,longitude = lon).data.flatten()
        
        corrs = np.zeros([len(lats),len(lons)])
        
        for i in range(len(lats)):
            for j in range(len(lons)):
                
                h2 = data2.sel(latitude = lats[i],longitude = lons[j]).data.flatten()
            
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
        lats,lons = np.meshgrid(lats,lons,indexing = indexings[split])
        lats = lats.flatten()
        lons = lons.flatten()
        
        corrs = np.reshape(corrs,len(lats),order = orders[split])
        
        return corrs,lats,lons

class heightenstime:
    
    def __init__(self,time,nmembers = 30,finedomain = False):
    
        '''
        Creates an ensemble of significant wave height for the given time,
        using all available model runs.
        '''
        
        # filepath format for netcdf files
        if finedomain:
            filefmt = '..//..//data//{:02d}//netcdf//ww3.sgom.202203{:02d}12.nc'
        else:
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
    
    def varplot(self,pos = None):
        
        '''
        Creates a plot of ensemble variance as a function of lead time
        '''
        
        V = self.ensemblevar()
        title = 'Time = ' + np.datetime_as_string(V.time[()],unit = 'm')
                                                 
        if pos is None:
            V = V.mean(dim = ('latitude','longitude'))
            title += ' (average over all gridpoints)'
        else:
            V = V.sel(latitude = pos[0],longitude = pos[1])
            title += ', Position = ({lat},{lon})'.format(lat = pos[0],lon = pos[1])
        
        # convert lead times to days
        lts = V.leadtime.astype('float64')/86400000000000
        
        # plot variance as a function of lead time
        fig,ax = plt.subplots()
        ax.plot(lts,V.data)
        ax.set_xlabel('Lead Time (days)')
        ax.set_ylabel('Wave Height Ensemble Variance ($m^2$)')
        ax.set_ylim(bottom = 0)
        ax.set_title(title)
    
    def varmaps(self):
        
        V = self.ensemblevar()
        for lt in V.leadtime:
            plt.figure()
            V.sel(leadtime = lt).plot()