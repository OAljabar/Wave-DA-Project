# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:38:45 2024

@author: owena
"""

import datetime as dt
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

class obsdata:
    
    def __init__(self,filepath,variable = 'hm0'):
        
        '''
        Create a data array containing a timeseries of observations
        '''
        
        # read data and split into entries
        datalist = open(filepath).read().splitlines()
        
        # get timestamps
        timefmt = '%Y-%m-%dT%H:%M:%SZ'
        times = [dt.datetime.strptime(entry[:20],timefmt) for entry in datalist[1:]]
        
        # dictionary containing indices for different variables (maybe automate)
        indices = {'hm0':slice(22,29)}
        
        var = [float(entry[indices[variable]]) for entry in datalist[1:]]
        data = xr.DataArray(var,coords = {'time':times})
        
        self.data = data
        self.lat = 18.41364
        self.lon = -93.7704
    
    def std(self,nbins = 10,plot = True,evenbins = False,reldif = False):
        
        '''
        Sorts the timeseries into bins based on magnitude and calculates the
        standard deviation of the variability of each bin. If plotting,
        includes a bar chart to show the size of each bin
        '''
        
        data = self.data.data
        
        # difference and mean of consecutive observations
        ddata = data[1:] - data[:-1]
        mdata = (data[1:] + data[:-1])/2
        if reldif:
            ddata = ddata/mdata
        
        # generate bin edges
        if evenbins:
            indices = np.linspace(0,len(ddata),nbins+1)
            remainders = indices - np.floor(indices)
            indices = (indices - remainders).astype('int')
            
            bins = np.zeros(nbins+1)
            bins[nbins] = max(mdata)
            mdsort = np.sort(mdata)
            for i in range(nbins):
                bins[i] = np.interp(remainders[i],[0,1],\
                                    [mdsort[indices[i]],mdsort[indices[i]+1]])
        else:
            bins = np.linspace(min(mdata),max(mdata),nbins+1)
        
        magnitude = np.zeros(nbins)
        std = np.zeros(nbins)
        binsize = np.zeros(nbins,dtype = 'int')
        
        for i in range(nbins):
            magnitude[i] = np.mean(mdata[(mdata > bins[i]) & (mdata < bins[i+1])])
            std[i] = np.std(ddata[(mdata > bins[i]) & (mdata < bins[i+1])])
            binsize[i] = len(mdata[(mdata > bins[i]) & (mdata < bins[i+1])])
        
        if plot:
            # initialise axes
            if evenbins:
                fig,ax = plt.subplots(figsize = (6,4))
                ax.set_title('Bin size = ' + str(binsize[0]))
            else:
                fig,[bax,ax] = plt.subplots(nrows = 2,figsize = (6,8))
            
                # bar chart to show bin size if uneven
                barwidth = min(bins[1:] - bins[:-1])
                bax.bar(bins[:-1],binsize,align = 'edge',width = barwidth)
                bax.set_ylabel('Frequency')
                bax.set_xlim([min(mdata),max(mdata)])
            
            # line plot of standard deviation
            ax.plot(magnitude,std,'-o')
            ax.set_xlabel('Wave Height (m)')
            ax.set_ylabel('Standard deviation')
            ax.set_xlim([min(mdata),max(mdata)])
            # plot line of best fit if difference is not normalised
            if not reldif:
                ax.set_ylabel('Standard deviation (m)')
                lmodel = stats.linregress(magnitude,std)
                slope = lmodel[0]
                intercept = lmodel[1]
                ax.axline((min(magnitude),slope*min(magnitude)+intercept),\
                          slope = slope,linestyle = '--',color = 'k')
                print(lmodel)
        
        return magnitude,std
    
    def ensdiff(self,run,nmembers = 30,finedomain = False,variable = 'hs',\
                interp = 'nearest'):
        
        if finedomain:
            filefmt = '..//..//data//{{:02d}}//netcdf//ww3.sgom.202203{:02d}12.nc'\
                .format(run)
        else:
            filefmt = '..//..//data//{{:02d}}//netcdf//ww3.gom.202203{:02d}12.nc'\
                .format(run)
        
        # get data for first ensemble member
        ens = xr.open_dataset(filefmt.format(1))[variable]
        
        if interp == 'nearest':
            
            #find indices for nearset neighbour interpolation to buoy location
            ilat = np.argmin(np.abs(ens.latitude.data - self.lat))
            ilon = np.argmin(np.abs(ens.longitude.data - self.lon))
            
            # get data from netcdf files
            ens = ens.isel(latitude = ilat,longitude = ilon)
            for member in range(2,nmembers+1):
                # get data from next member and add to array
                nextmember = xr.open_dataset(filefmt.format(member))[variable]\
                    .isel(latitude = ilat,longitude = ilon)
                ens = xr.concat([ens,nextmember],dim = 'member')
        else:
            
            # get data and interpolate to buoy location
            ens = ens.interp(latitude = self.lat,longitude = self.lon)
            for member in range(2,nmembers+1):
                # get data from next member and add to array
                nextmember = xr.open_dataset(filefmt.format(member))[variable]\
                    .interp(latitude = self.lat,longitude = self.lon)
                ens = xr.concat([ens,nextmember],dim = 'member')
        
        # assign coordinates to ensemble members
        ens = ens.assign_coords(member = range(1,nmembers+1))
        
        ensmean = ens.mean(dim = 'member')
        diff = self.data - ensmean
        
        fig,ax = plt.subplots()
        ax.scatter(ensmean.sel(time = slice(min(diff.time),max(diff.time))).data,\
                    diff.data,marker = 'x')
        ax.axhline(color = 'k',linestyle = '--')
        ax.set_title(variable + ' (m)')
        ax.set_xlabel('Ensemble mean')
        ax.set_ylabel('Observation - ensemble mean')
        
        return diff