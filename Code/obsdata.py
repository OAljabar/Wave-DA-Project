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
import re

class getobs:
    
    def __init__(self,filepath,variables = ['HM0']):
        
        '''
        Create a data array containing a timeseries of observations
        '''
        
        # read data and split into entries
        datalist = open(filepath).read().splitlines()
        
        # get indices for each variable and store in dictionary
        indices = {}
        index = 0
        varlist= re.split(r'(\W+)',datalist[0])
        for i in range(len(varlist)-1):
            if varlist[i][0] != ' ':
                entry = varlist[i]
                length = len(varlist[i]) + len(varlist[i+1])
                indices[entry] = slice(index,index+length)
                index += length
        
        # get timestamps
        timefmt = '%Y-%m-%dT%H:%M:%SZ'
        times = [dt.datetime.strptime(entry[indices['TIMESTAMP']],timefmt) for entry in datalist[1:]]
        
        starti = 0
        endi = len(times)
        
        ds = xr.Dataset()
        for variable in variables:
            var = [float(entry[indices[variable]]) for entry in datalist[1:]]
            data = xr.DataArray(var,coords = {'time':times})
            
            # select range to exclude missing values
            endi = min(endi,next(i for i in reversed(range(len(data))) if data[i].data != data[0].data))
            starti = max(starti,next(i for i in reversed(range(endi)) if data[i].data == data[0].data) + 1)
            
            ds[variable] = data
        
        ds = ds.sel(time = slice(times[starti],times[endi]))
        
        self.data = ds
        self.lat = 18.41364
        self.lon = -93.7704
        self.variables = variables
        self.varnames = {'HM0':'hs'}

class obsplots(getobs):

    def plotvars(self,variables,units,varname,ax = None,times = None):
        
        if ax is None:
            fig,ax = plt.subplots()
        for variable in variables:
            ax.plot(self.data.time.sel(time = times),\
                    self.data[variable].sel(time = times),'k',label = variable)
            #self.data[variable].sel(time = times).plot(ax = ax,label = variable)
        ax.set_ylabel(varname + ' (' + units + ')')
        ax.set_xlabel('Time')
        if len(variables) > 1:
            ax.legend()
    
    def std(self,nbins = 10,plot = True,evenbins = False,reldif = False,\
            xvar = 'HM0',yvar = 'HM0',fitline = True,ax = None):
        
        '''
        Sorts the timeseries into bins based on magnitude and calculates the
        standard deviation of the variability of each bin. If plotting,
        includes a bar chart to show the size of each bin
        '''
        
        # mean and difference of consecutive observations for x and y axes
        xdata = (self.data[xvar].data[1:] + self.data[xvar].data[:-1])/2
        ydata = self.data[yvar].data[1:] - self.data[yvar].data[:-1]
        
        if reldif:
            ydata = ydata/xdata
        
        # generate bin edges
        if evenbins:
            indices = np.linspace(0,len(ydata),nbins+1)
            remainders = indices - np.floor(indices)
            indices = (indices - remainders).astype('int')
            
            bins = np.zeros(nbins+1)
            bins[nbins] = max(xdata)
            xdsort = np.sort(xdata)
            for i in range(nbins):
                bins[i] = np.interp(remainders[i],[0,1],\
                                    [xdsort[indices[i]],xdsort[indices[i]+1]])
        else:
            bins = np.linspace(min(xdata),max(xdata),nbins+1)
        
        magnitude = np.zeros(nbins)
        lintervals = np.zeros(nbins)
        hintervals = np.zeros(nbins)
        std = np.zeros(nbins)
        binsize = np.zeros(nbins,dtype = 'int')
        
        for i in range(nbins):
            # select data from appropriate bin
            binindices = (xdata > bins[i]) & (xdata < bins[i+1])
            binsize[i] = len(xdata[binindices])
            if binsize[i] > 1:
                
                # calculate statistics from bin data
                magnitude[i] = np.mean(xdata[binindices]) # mean
                res = stats.bootstrap((xdata[binindices],),np.mean)
                lintervals[i] = res.confidence_interval[0]
                hintervals[i] = res.confidence_interval[1]
                std[i] = np.std(ydata[binindices])
                
        # remove data from empty bins
        magnitude = magnitude[binsize > 1]
        lintervals = lintervals[binsize > 1]
        hintervals = hintervals[binsize > 1]
        std = std[binsize > 1]
        bins = bins[np.concatenate((binsize > 1,[True]))]
        binsize = binsize[binsize > 1]
        
        xerr = [magnitude - lintervals,hintervals - magnitude]
        
        if plot:
            # initialise axes
            if evenbins:
                if ax is None:
                    fig,ax = plt.subplots(figsize = (6,4))
                ax.set_title('Bin size = ' + str(binsize[0]))
            else:
                fig,[bax,ax] = plt.subplots(nrows = 2,figsize = (6,8))
            
                # bar chart to show bin size if uneven
                barwidth = min(bins[1:] - bins[:-1])
                bax.bar(bins[:-1],binsize,align = 'edge',width = barwidth)
                bax.set_ylabel('Frequency')
                bax.set_xlim([min(xdata),max(xdata)])
            
            # line plot of standard deviation
            ax.errorbar(magnitude,std,xerr = xerr,fmt = '-o')
            ax.set_xlabel(xvar)
            ax.set_ylabel(yvar + ' standard deviation')
            ax.set_xlim([min(xdata),max(xdata)])
            
            # plot line of best fit if difference is not normalised
            if fitline:
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
    
    def ensmeandiff(self,plot = None,variable = 'HM0',plotsettings = {}):
        
        # get ensemble forecast data if not already saved (RESTRUCTURE THIS)
        try:
            ens = self.ensdata
        except:
            ensvariable = self.varnames[variable]
            
            ens = self.getensdata(variable = ensvariable,interp = 'nearest')
        
        ensmean = ens.mean(dim = 'member')
        diff = self.data[variable] - ensmean
        ensmean = ensmean.sel(time = slice(min(diff.time),max(diff.time)))
        time = (diff.time.data - diff.time.data[0]).astype('float')/3.6e12
        
        with plt.rc_context(plotsettings):
        
            if plot == 'scatter' or plot == 'both':
                fig,ax = plt.subplots()
                scatter = ax.scatter(ensmean.data,diff.data,marker = 'x',\
                           c = time,cmap = 'spring')
                cbar = plt.colorbar(scatter)
                cbar.set_label('Leadtime (h)')
                ax.axhline(color = 'k',linestyle = '--')
                ax.set_title(self.varnames[variable] + ' (m)')
                ax.set_xlabel('Ensemble mean')
                ax.set_ylabel('Observation - ensemble mean')
            if plot == 'hist' or plot == 'both':
                fig,ax = plt.subplots()
                ax.hist(diff)
                ax.axvline(color = 'k',linestyle = '--')
                ax.set_title(self.varnames[variable] + ' (m)')
                ax.set_xlabel('Observation - ensemble mean')
                ax.set_ylabel('Frequency')
        
        return diff,ensmean
    
    def ominusb(self,ax = None):
        
        if not hasattr(self,'fulldiff'):
            diff = xr.DataArray()
            
            for i in range(6,13):
                self.getensdata(i,30,True);
                newdiff,_ = self.ensmeandiff()
                newdiff = newdiff.rename({'time':'leadtime'})
                newdiff = newdiff.assign_coords(leadtime = range(169))
                diff = xr.concat((diff,newdiff),dim = 'run')
            
            self.fulldiff = diff.isel(run = slice(1,7)).assign_coords(run = range(6,13))
        
        if ax is None:
            fig,ax = plt.subplots(figsize = (9,6))
        
        ax.hist(self.fulldiff.data.flatten())
    
    def ensdiff(self,variable = 'HM0',unit = 'm'):
        
        ens = self.ensdata
        ensmean = ens.mean(dim = 'member')
        
        diff = self.data[variable] - ens
        
        sqdiff = diff*diff
        rmse = np.sqrt(sqdiff.mean(dim = 'member'))
        time = (diff.time.data - diff.time.data[0]).astype('float')/3.6e12
        
        fig,ax = plt.subplots()
        scatter = ax.scatter(ensmean,rmse,marker = 'x',\
                   c = time,cmap = 'spring')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Leadtime (h)')
        ax.set_xlabel(variable + ' ensemble mean (' + unit + ')')
        ax.set_ylabel(variable + ' rmse (' + unit + ')')
        
        return diff
    
    def peakerror(self,variable = 'HM0'):
        
        # get ensemble forecast data if not already saved (RESTRUCTURE THIS)
        try:
            ens = self.ensdata
        except:
            ensvariable = self.varnames[variable]
            
            ens = self.getensdata(variable = ensvariable)
        
        self.data = self.data.sel(time = ens.time)
        obspeakindex = np.argmax(self.data[variable].data)
        obspeaktime = self.data.time.data[obspeakindex]
        
        enspeakindices = [np.argmax(ens.sel(member = i).data) for i in ens.member]
        enspeaktimes = np.array([ens.time.data[i] for i in enspeakindices])
        
        peakerrors = ((enspeaktimes - obspeaktime)/3.6e12).astype('int')
        
        plt.hist(peakerrors,bins = range(min(peakerrors),max(peakerrors)+2),\
                 align = 'left')
        
        return peakerrors
        return obspeaktime,enspeaktimes
        