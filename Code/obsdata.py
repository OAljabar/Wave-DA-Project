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
        self.variable = variable
        self.varnames = {'hm0':'hs'}
    
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
    
    def getensdata(self,run = None,nmembers = None,finedomain = None,variable = 'hs',\
                interp = 'nearest'):
        
        if run is None:
            run = int(input('Select model run: \n'))
        if nmembers is None:
            nmembers = int(input('Select number of ensemble members: \n'))
        if finedomain is None:
            finedomain = bool(input('Use data from inner domain? (True/False): \n'))
        
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
        
        self.ensdata = ens
        return ens
    
    def ensdiff(self,plot = None):
        
        # get ensemble forecast data if not already saved (RESTRUCTURE THIS)
        try:
            ens = self.ensdata
        except:
            variable = self.varnames[self.variable]
            
            ens = self.getensdata(variable = variable,interp = 'nearest')
        
        ensmean = ens.mean(dim = 'member')
        diff = self.data - ensmean
        ensmean = ensmean.sel(time = slice(min(diff.time),max(diff.time)))
        time = (diff.time.data - diff.time.data[0]).astype('float')/3.6e12
        
        if plot == 'scatter' or plot == 'both':
            fig,ax = plt.subplots()
            scatter = ax.scatter(ensmean.data,diff.data,marker = 'x',\
                       c = time,cmap = 'spring')
            cbar = plt.colorbar(scatter)
            cbar.set_label('Time since start of run (h)')
            ax.axhline(color = 'k',linestyle = '--')
            ax.set_title(self.varnames[self.variable] + ' (m)')
            ax.set_xlabel('Ensemble mean')
            ax.set_ylabel('Observation - ensemble mean')
        if plot == 'hist' or plot == 'both':
            fig,ax = plt.subplots()
            ax.hist(diff)
            ax.axvline(color = 'k',linestyle = '--')
            ax.set_title(self.varnames[self.variable] + ' (m)')
            ax.set_xlabel('Observation - ensemble mean')
            ax.set_ylabel('Frequency')
        
        return diff,ensmean
    
    def peakerror(self):
        
        # get ensemble forecast data if not already saved (RESTRUCTURE THIS)
        try:
            ens = self.ensdata
        except:
            run = int(input('Select model run: \n'))
            nmembers = int(input('Select number of ensemble members: \n'))
            finedomain = bool(input('Use data from inner domain? (True/False): \n'))
            variable = self.varnames[self.variable]
            
            ens = self.getensdata(run,nmembers,finedomain,variable,'nearest')
        
        obspeakindex = np.argmax(self.data.data)
        obspeaktime = self.data.time.data[obspeakindex]
        
        enspeakindices = [np.argmax(ens.sel(member = i).data) for i in ens.member]
        enspeaktimes = np.array([ens.time.data[i] for i in enspeakindices])
        
        peakerrors = ((enspeaktimes - obspeaktime)/3.6e12).astype('int')
        
        plt.hist(peakerrors,bins = range(min(peakerrors),max(peakerrors)+2),\
                 align = 'left')
        
        return peakerrors
        return obspeaktime,enspeaktimes
        