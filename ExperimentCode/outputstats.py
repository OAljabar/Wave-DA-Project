#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:14:25 2024

@author: mg838076
"""

import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import cartopy.crs as ccrs

class outputdata:
    
    '''
    Loads data from DA system output, with methods for analysis. Assumes a 
    filename format of "step{t}member{n}{label}.nc" with labels being "da", "no"
    or truth. Ensemble member 0 should be truth and ensemble members for da and
    no should be enumerated 1,2,...,nmembers.
    '''
    
    def __init__(self,outputdir):
        
        '''
        Sorts files into DA, no DA and truth and gets ensemble size and timesteps
        '''
        
        self.outputdir = outputdir
        self.files = os.listdir(outputdir)
        
        self.dafiles = []
        self.nofiles = []
        self.truthfiles = []
        
        for file in self.files:
            if not file[-3:] == '.nc':
                self.files.remove(file)
                continue
            if 'da' in file:
                self.dafiles.append(file)
            if 'no' in file:
                self.nofiles.append(file)
            if 'truth' in file:
                self.truthfiles.append(file)
                self.files.remove(file)
            
        self.nmembers = max([int(file[12:14]) for file in self.files])
        self.tinit = min([int(file[4:6]) for file in self.files])
        self.tfinal = max([int(file[4:6]) for file in self.files])
        self.nt = self.tfinal - self.tinit + 1
        
        self.dafiles.sort()
        self.nofiles.sort()
        self.truthfiles.sort()
    
    def readtruth(self):
        
        '''
        Reads truth files into xarray dataset
        '''
        
        truthdata = xr.open_dataset(self.outputdir + self.truthfiles[0])
        
        self.truthdata = truthdata
    
    def set_filefmt(self,filefmt = 'step{:02d}member{:02d}'):
        self.filefmt = filefmt
        
    def readall(self):
        
        for label in ['da','no','truth']:
            self.readfiles(label)
        #self.readtruth()
        
    def readfiles(self,label):
        
        '''
        Reads files associated with specified label (da, no or truth) into dataset
        '''
        
        if label == 'da':
            filelist = self.dafiles
        elif label == 'no':
            filelist = self.nofiles
        elif label == 'truth':
            filelist = self.truthfiles
        else:
            raise Exception('Valid labels are "da", "no" and "truth"')
        
        if not hasattr(self,'filefmt'):
            self.set_filefmt()
        filefmt = self.outputdir + self.filefmt + label + '.nc'
        
        if label == 'truth':
            data = xr.open_dataset(filefmt.format(self.tinit,0))
        else:
            data = xr.open_dataset(filefmt.format(self.tinit,1))
            for i in range(2,self.nmembers+1):
                nextmember = xr.open_dataset(filefmt.format(self.tinit,i))
                data = xr.concat([data,nextmember],dim = 'member')
        
        for t in range(self.tinit+1,self.tfinal+1):
            if label == 'truth':
                nextdata = xr.open_dataset(filefmt.format(t,0))
            else:
                nextdata = xr.open_dataset(filefmt.format(t,1))
                for i in range(2,self.nmembers+1):
                    nextmember = xr.open_dataset(filefmt.format(t,i))
                    nextdata = xr.concat([nextdata,nextmember],dim = 'member')
            data = xr.concat([data,nextdata],dim = 'step')
            
        if not label == 'truth':
            data = data.assign_coords(member = range(1,self.nmembers + 1))
        data = data.assign_coords(step = range(self.tinit,self.tfinal+1))
        if label == 'da':
            self.dadata = data
        elif label == 'no':
            self.nodata = data
        elif label == 'truth':
            self.truthdata = data 
        
class outputstats:
    
    def __init__(self,outputdata):
        
        self.dadata = outputdata.dadata
        self.nodata = outputdata.nodata
        self.truthdata = outputdata.truthdata
        self.nmembers = outputdata.nmembers
        self.nt = outputdata.nt
    
    def timeseries(self,lat,lon,variable,step):
        
        coords = {'latitude':lat,'longitude':lon,'step':step}
        dadata = self.dadata[variable].sel(coords).mean(dim = 'member')
        nodata = self.nodata[variable].sel(coords).mean(dim = 'member')
        truthdata = self.truthdata[variable].sel(coords)
        
        time = truthdata.leadtime.data
        
        fig,ax = plt.subplots(figsize = (10,5))
        ax.plot(time,truthdata.data,color = 'k',label = 'Truth')
        ax.plot(time,dadata.data,color = 'tab:orange',label = 'DA')
        ax.plot(time,nodata.data,color = 'tab:blue',label = 'No DA')
        ax.legend()
        ax.set_xlabel('Leadtime')
        ax.set_ylabel(variable)
    
    def plume(self,lat,lon,variable,step = None,nmembers = None):
        
        coords = {'latitude':lat,'longitude':lon}
        if self.nt > 1:
            if step is None:
                step = input('Multiple timesteps available. Specify step:\n')
            coords['step'] = step
        dadata = self.dadata[variable].sel(coords)
        nodata = self.nodata[variable].sel(coords)
        truthdata = self.truthdata[variable].sel(coords)
        
        if nmembers is None:
            nmembers = self.nmembers
        
        obstime = truthdata.leadtime.data
        forecasttime = nodata.leadtime.data
        
        fig,ax = plt.subplots(figsize = (10,5))
        ax.plot(obstime,truthdata.data,color = 'k',label = 'Truth')
        ax.plot(forecasttime,dadata.isel(member = 0),color = 'tab:orange',label = 'DA')
        ax.plot(forecasttime,nodata.isel(member = 0),color = 'tab:blue',label = 'No DA')
        for i in range(1,nmembers):
            ax.plot(forecasttime,dadata.isel(member = i),color = 'tab:orange')
            ax.plot(forecasttime,nodata.isel(member = i),color = 'tab:blue')
        ax.set_xlabel('Leadtime (days)')
        ax.set_ylabel(variable + ' (m)')
        ax.set_title('Variable: ' + variable + ', Position: ({:},{:})'.format(lat,lon))
        ax.legend()
    
    def rmse(self,lat,lon,variable,step = None,normalise = False,plot = True,savefig = False,filename = None):
        
        coords = {'latitude':lat,'longitude':lon}
        if self.nt > 1:
            if step is None:
                step = input('Multiple timesteps available. Specify step:\n')
            coords['step'] = step
        dadata = self.dadata[variable].sel(coords)
        nodata = self.nodata[variable].sel(coords)
        truthdata = self.truthdata[variable].sel(coords)
        
        dadiff = dadata - truthdata
        darmse = np.sqrt((dadiff * dadiff).mean(dim = 'member'))
        nodiff = nodata - truthdata
        normse = np.sqrt((nodiff * nodiff).mean(dim = 'member'))
        
        if normalise:
            darmse = darmse/truthdata
            normse = normse/truthdata
        
        if plot:
            fig,ax = plt.subplots(figsize = (8,5))
            ax.plot(darmse.leadtime,darmse.data,'k--',label = 'DA')
            ax.plot(normse.leadtime,normse.data,'k',label = 'No DA')
            ax.set_xlabel('Leadtime (days)')
            ax.set_ylabel('Ensemble RMS error (m)')
            ax.set_title('Variable: ' + variable + ', Position: ({:},{:}), Timestep: {:}'.format(lat,lon,step))
            ax.axvline(3,ls = ':')
            ax.legend()
            
            if savefig:
                fig.savefig(filename)
        
        return darmse,normse
    
    def rmsebyday(self,step,lat,lon,variable):
        
        # calculate ensemble means at given location
        dadata = self.dadata[variable].sel(latitude = lat,longitude = lon).mean(dim = 'member')
        nodata = self.nodata[variable].sel(latitude = lat,longitude = lon).mean(dim = 'member')
        
        truthdata = self.truthdata[variable].sel(latitude = lat,longitude = lon)
        
        dadiff = dadata - truthdata
        nodiff = nodata - truthdata
        
        # find available leadtimes
        lts = np.arange(max(0,step - np.max(dadiff.step)),min(7,step))
        
        darmse = [np.sqrt((dadiff.sel(step = step - lt,leadtime = slice(lt,lt+1))**2).mean())\
                  for lt in lts]
        normse = [np.sqrt((nodiff.sel(step = step - lt,leadtime = slice(lt,lt+1))**2).mean())\
                  for lt in lts]
            
        fig,ax = plt.subplots()
        ax.plot(lts+0.5,darmse,'k:',label = 'DA')
        ax.plot(lts+0.5,normse,'k',label = 'No DA')
        ax.set_xlabel('Leadtime (days)')
        ax.set_ylabel('RMS error in ensemble mean')
        datestring = '{:02d}/03/2022T12:00:00'.format(step+6)
        ax.set_title(f'Position: ({lat},{lon}), Variable: '+variable+\
                     ', 24hr period starting '+datestring)
        ax.legend()
        
        return darmse,normse
    
    def rmseallruns(self,lat,lon,variable,dt = 0.25):
        
        coords = {'step':slice(1,self.nt),'latitude':lat,'longitude':lon}
        
        dadata = self.dadata[variable].sel(coords).mean(dim = 'member')
        nodata = self.nodata[variable].sel(coords).mean(dim = 'member')
        truthdata = self.truthdata[variable].sel(coords)
        
        dadiff = dadata - truthdata
        nodiff = nodata - truthdata
        
        times = np.arange(min(truthdata.leadtime.data),max(truthdata.leadtime.data),dt)
        
        darmse = [np.sqrt((dadiff.sel(leadtime = slice(t,t+dt))**2).mean().data[()]) for t in times]
        normse = [np.sqrt((nodiff.sel(leadtime = slice(t,t+dt))**2).mean().data[()]) for t in times]
        
        return darmse,normse
    
    def peakerrors(self,step,lat,lon,variable,lt):
        
        coords = {'step':step-lt,'latitude':lat,'longitude':lon,\
                  'leadtime':slice(lt,lt+1)}
        
        dadata = self.dadata[variable].sel(coords)
        
        nodata = self.nodata[variable].sel(coords)
        
        truthdata = self.truthdata[variable].sel(coords)
        truthi = np.argmax(truthdata.data)
        truthmag = truthdata.data[truthi]
        
        daindices = np.array([np.argmax(dadata.isel(member = i).data) for i in range(self.nmembers)])
        noindices = np.array([np.argmax(nodata.isel(member = i).data) for i in range(self.nmembers)])
        damag = np.array([dadata.isel(member = i).data[index] for i,index in enumerate(daindices)])
        nomag = np.array([nodata.isel(member = i).data[index] for i,index in enumerate(noindices)])
        
        fig,ax =plt.subplots()
        ax.scatter(daindices,damag,marker = 'x',color = 'tab:orange',label = 'DA')
        ax.scatter(noindices,nomag,marker = 'x',color = 'tab:blue',label = 'No DA')
        with plt.rc_context({'lines.markersize':10}):
            ax.scatter(truthi,truthmag,marker = 'x',color = 'k',label = 'Truth')
        ax.legend()
        datestring = '{:02d}/03/2022T12:00:00'.format(step+6)
        ax.set_title(f'Position: ({lat},{lon}), Variable: '+variable+\
                     ', 24hr period starting '+datestring)
        
        rmses = {'datiming':rms(daindices - truthi),'notiming':rms(noindices-truthi),\
                   'damagnitude':rms(damag - truthmag),'nomagnitude':rms(nomag - truthmag)}
        
        return rmses
    
    def rmsereduction(self,lat,lon,variable,step = None,normalise = False):
        
        # calculate rmse with and without DA
        darmse,normse = self.rmse(lat,lon,variable,step,normalise,False)
        
        # calculate percentage reduction from DA
        reduction = 100*(1 - darmse/normse)
        
        fig,ax = plt.subplots(figsize = (8,5))
        ax.plot(reduction.leadtime,reduction.data,'k')
        ax.set_xlabel('Leadtime (days)')
        ax.set_ylabel('RMS error reduction (%)')
        ax.set_title('Variable: ' + variable + ', Position: ({:},{:})'.format(lat,lon) + \
                     ', Run: {:02d}/03/2022'.format(step+6))
        ax.axhline(0,ls =  '--',color = 'k')
        ax.axvline(3,ls = ':')
        
        return reduction
        
    
    def incrementmap(self,leadtime,variable,units,member = None,step = None,normalise = False):
        
        coords = {'leadtime':leadtime}
        if self.nt > 1:
            if step is None:
                raise Exception('Multiple timesteps available. Step must be specified')
            coords['step'] = step
        dadata = self.dadata[variable].sel(coords)
        nodata = self.nodata[variable].sel(coords)
        if member is None:
            dadata = dadata.mean(dim = 'member')
            nodata = nodata.mean(dim = 'member')
        else:
            dadata = dadata.sel(member = member)
            nodata = nodata.sel(member = member)
        
        increments = dadata - nodata
        if normalise:
            truthdata = self.truthdata[variable].sel(coords)
            innovations = truthdata - nodata
            increments = increments/innovations
        
        fig,ax = plt.subplots(figsize = (7,5),subplot_kw={'projection':ccrs.PlateCarree()})
        ax.coastlines()
        mesh = ax.pcolormesh(increments.longitude,increments.latitude,increments.data,\
                      norm = colors.CenteredNorm(),cmap = 'PRGn')
        cbar = plt.colorbar(mesh)
        cbar.set_label(variable+' increment ('+units+')')
        ax.set_title('Leadtime: ' + str(leadtime) + ' days, variable: ' + variable)
        
        return increments
    
    def incinnseries(self,lat,lon,variable,member = None,step = None):
        
        coords = {'latitude':lat,'longitude':lon}
        if self.nt > 1:
            if step is None:
                raise Exception('Multiple timesteps available. Step must be specified')
            coords['step'] = step
        truthdata = self.truthdata[variable].sel(coords)
        dadata = self.dadata[variable].sel(coords)
        nodata = self.nodata[variable].sel(coords)
        if member is None:
            dadata = dadata.mean(dim = 'member')
            nodata = nodata.mean(dim = 'member')
        else:
            dadata = dadata.sel(member = member)
            nodata = nodata.sel(member = member)
        
        increments = dadata - nodata
        innovations = truthdata - nodata
        
        fig,ax = plt.subplots(figsize = (10,5))
        increments.plot.line('g',ax = ax,label = 'Increments')
        innovations.plot.line('r',ax = ax,label = 'Innovations')
        ax.axhline(y = 0,ls = '--',color = 'k')
        ax.set_xlabel('Leadtime (days)')
        ax.set_ylabel('Hs increment (m)')
        ax.set_title('Position: ({:},{:})'.format(lat,lon))
        ax.legend()
    
    def rmsemap(self,step,variable,units,ltstart = None,ltstop = None):
        
        coords = {'step':step,'leadtime':slice(ltstart,ltstop)}
        
        dadata = self.dadata[variable].sel(coords).mean(dim = 'member')
        nodata = self.nodata[variable].sel(coords).mean(dim = 'member')
        
        truthdata = self.truthdata[variable].sel(coords)
        
        dadiff = dadata - truthdata
        nodiff = nodata - truthdata
        
        darmse = np.sqrt((dadiff*dadiff).mean(dim = 'leadtime'))
        normse = np.sqrt((nodiff*nodiff).mean(dim = 'leadtime'))
        reduction = 100*(1 - darmse/normse)
        
        fig = plt.figure(figsize = (13,5))
        dax = fig.add_subplot(121,projection = ccrs.PlateCarree())
        nax = fig.add_subplot(122,projection = ccrs.PlateCarree())
        damesh = dax.pcolormesh(darmse.longitude,darmse.latitude,reduction.data,\
                       vmin = -100,vmax = 100,cmap = 'RdBu')
        nomesh = nax.pcolormesh(normse.longitude,normse.latitude,normse.data,\
                       cmap = 'Reds')
        dacbar = plt.colorbar(damesh)
        dacbar.set_label('RMS error reduction(%)')
        nocbar = plt.colorbar(nomesh)
        nocbar.set_label('RMS error ('+units+')')
        dax.coastlines()
        nax.coastlines()
        fig.suptitle('RMS error in '+variable+f' ensemble mean (day {ltstart} to day {ltstop})')
        dax.set_title('DA')
        nax.set_title('No DA')
    
    def spreadseries(self,lat,lon,variable,units,step):
        
        coords = {'step':step,'latitude':lat,'longitude':lon}
        
        dadata = self.dadata[variable].sel(coords)
        nodata = self.nodata[variable].sel(coords)
        
        daspread = dadata.std(dim = 'member')
        nospread = nodata.std(dim = 'member')
        
        fig,ax = plt.subplots(figsize = (8,5))
        ax.plot(daspread.leadtime,daspread.data,'k--',label = 'DA')
        ax.plot(nospread.leadtime,nospread.data,'k',label = 'No DA')
        ax.set_xlabel('Leadtime (days)')
        ax.set_ylabel('Ensemble standard deviation ('+units+')')
        ax.set_title('Variable: ' + variable + ', Position: ({:},{:}), Timestep: {:}'.format(lat,lon,step))
        ax.axvline(3,ls = ':')
        ax.legend()
        
        return daspread,nospread
    
    def spreadmap(self,step,variable,units,ltstart,ltstop):
        
        coords = {'step':step,'leadtime':slice(ltstart,ltstop)}
        
        if ltstart == ltstop:
            tstring = f'(leadtime {ltstart} days)'
        else:
            tstring = f'(averaged over leadtime {ltstart} to {ltstop} days)'
        
        daspread = self.dadata[variable].sel(coords).std(dim = 'member').mean(dim = 'leadtime')
        nospread = self.nodata[variable].sel(coords).std(dim = 'member').mean(dim = 'leadtime')

        reduction = 100*(1 - daspread/nospread)
        
        fig = plt.figure(figsize = (13,5))
        dax = fig.add_subplot(121,projection = ccrs.PlateCarree())
        nax = fig.add_subplot(122,projection = ccrs.PlateCarree())
        damesh = dax.pcolormesh(daspread.longitude,daspread.latitude,reduction.data,\
                       vmin = -100,vmax = 100,cmap = 'RdBu')
        nomesh = nax.pcolormesh(nospread.longitude,nospread.latitude,nospread.data,\
                       cmap = 'Reds')
        dacbar = plt.colorbar(damesh)
        dacbar.set_label('Ensemble spread reduction(%)')
        nocbar = plt.colorbar(nomesh)
        nocbar.set_label('RMS error ('+units+')')
        dax.coastlines()
        nax.coastlines()
        fig.suptitle(variable+' ensemble standard deviation '+tstring)
        dax.set_title('DA')
        nax.set_title('No DA')
    
    def rmsespread(self,lat,lon,variable,step):
        
        coords = {'step':step,'latitude':lat,'longitude':lon}
        
        dadata = self.dadata[variable].sel(coords)
        truthdata = self.truthdata[variable].sel(coords)
        
        diff = dadata - truthdata
        rmse = np.sqrt((diff*diff).mean(dim = 'member'))
        spread = dadata.std(dim = 'member')
        
        fig,ax = plt.subplots(figsize = (10,5))
        ax.plot(rmse.leadtime,rmse.data,label = 'Error')
        ax.plot(spread.leadtime,spread.data,label = 'Spread')
        ax.legend()
        
def rms(data):
    
    '''
    Calculates the root mean square of given data
    '''
    
    return np.sqrt(np.mean(data*data))