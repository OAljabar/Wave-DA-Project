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
        self.hsdafiles = []
        self.nofiles = []
        self.truthfiles = []
        
        for file in self.files:
            if not file[-3:] == '.nc':
                self.files.remove(file)
                continue
            if 'da' in file:
                if 'hs' in file:
                    self.hsdafiles.append(file)
                else:
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
        self.hsdafiles.sort()
        self.nofiles.sort()
        self.truthfiles.sort()
        
        self.labels = ['da','hsda','no','truth']
    
    def set_filefmt(self,filefmt = 'step{:02d}member{:02d}'):
        self.filefmt = filefmt
        
    def readall(self,labels = None):
        
        if labels is None:
            labels = self.labels
        
        for label in labels:
            self.readfiles(label)
        
    def readfiles(self,label):
        
        '''
        Reads files associated with specified label (da, no or truth) into dataset
        '''
        
        if label not in self.labels:
            raise Exception('Valid labels are "da", "hsda", "no" and "truth"')
        
        if not hasattr(self,'filefmt'):
            filefmt = input('Specify file format. Default is step{:02d}member{:02d}\n')
            if filefmt == '':
                self.set_filefmt()
            else:
                self.set_filefmt(filefmt)
                
        filefmt = self.outputdir + self.filefmt + label + '.nc'
        
        if label == 'truth':
            data = xr.open_dataset(filefmt.format(self.tinit,0)).astype('float16')
        else:
            data = xr.open_dataset(filefmt.format(self.tinit,1)).astype('float16')
            for i in range(2,self.nmembers+1):
                nextmember = xr.open_dataset(filefmt.format(self.tinit,i)).astype('float16')
                data = xr.concat([data,nextmember],dim = 'member')
        
        print(label+'data loaded for step 0')
        
        for t in range(self.tinit+1,self.tfinal+1):
            if label == 'truth':
                nextdata = xr.open_dataset(filefmt.format(t,0)).astype('float16')
            else:
                nextdata = xr.open_dataset(filefmt.format(t,1)).astype('float16')
                for i in range(2,self.nmembers+1):
                    nextmember = xr.open_dataset(filefmt.format(t,i)).astype('float16')
                    nextdata = xr.concat([nextdata,nextmember],dim = 'member')
            data = xr.concat([data,nextdata],dim = 'step')
            
            print(label+f'data loaded for step {t}')
            
        if not label == 'truth':
            data = data.assign_coords(member = range(1,self.nmembers + 1))
        data = data.assign_coords(step = range(self.tinit,self.tfinal+1))
        if label == 'da':
            self.dadata = data
        if label == 'hsda':
            self.hsdadata = data
        elif label == 'no':
            self.nodata = data
        elif label == 'truth':
            self.truthdata = data 
        
class outputstats:
    
    def __init__(self,outputdata):
        
        self.dadata = outputdata.dadata
        self.hsdadata = outputdata.hsdadata
        self.nodata = outputdata.nodata
        self.truthdata = outputdata.truthdata
        self.nmembers = outputdata.nmembers
        self.nt = outputdata.nt
        self.varnames = {'hs':'Hs','t01':'Tm01','t02':'Tm02'}
        self.units = {'hs':' (m)','t01':' (s)','t02':' (s)'}
        
    def select(self,variable,coords = None,ensstat = xr.DataArray.mean):
    
        dadata = self.dadata[variable].sel(coords)
        hsdadata = self.hsdadata[variable].sel(coords)
        nodata = self.nodata[variable].sel(coords)
        
        if ensstat is not None:
            dadata = ensstat(dadata,dim = 'member')
            hsdadata = ensstat(hsdadata,dim = 'member')
            nodata = ensstat(nodata,dim = 'member')
        
        # remove 'member' from coords if present and select truth data
        coords.pop('member',None)
        truthdata = self.truthdata[variable].sel(coords)
        
        return dadata,hsdadata,nodata,truthdata
    
    def timeseries(self,data,styles,xlabel,ylabel,title = '',x = None,\
                   ax = None,figsize = (10,5),withlabels = True):
        
        if ax is None:
            fig,ax = plt.subplots(figsize = figsize)
        
        for label in data.keys():
            if x is None:
                data[label].plot.line(styles[label],ax = ax,\
                                      label = label if withlabels else None)
            else:
                ax.plot(x,data[label],styles[label],\
                        label = label if withlabels else None)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    
    def rmse(self,step,lat,lon,variable,normalise = False):
        
        coords = {'latitude':lat,'longitude':lon,'step':step}

        dadata,hsdadata,nodata,truthdata = self.select(variable,coords,ensmean = False)
        
        dadiff = dadata - truthdata
        darmse = np.sqrt((dadiff * dadiff).mean(dim = 'member'))
        hsdadiff = hsdadata - truthdata
        hsdarmse = np.sqrt((hsdadiff * hsdadiff).mean(dim = 'member'))
        nodiff = nodata - truthdata
        normse = np.sqrt((nodiff * nodiff).mean(dim = 'member'))
        
        if normalise:
            darmse = darmse/truthdata
            hsdarmse = hsdarmse/truthdata
            normse = normse/truthdata
        
        return darmse,hsdarmse,normse
    
    def ensmeanseries(self,step,lat,lon,variable):
        
        coords = {'latitude':lat,'longitude':lon,'step':step}
        dadata,hsdadata,nodata,truthdata = self.select(variable,coords)
        
        data = {'Truth':truthdata,'DA':dadata,'Hs DA':hsdadata,'No DA':nodata}
        styles = {'Truth':'k','DA':'g','Hs DA':'b','No DA':'r'}
        
        self.timeseries(data,styles,'Leadtime (days)',\
                        self.varnames[variable]+self.units[variable])
    
    def plume(self,lat,lon,variable,step,nmembers = None):
        
        if nmembers is None:
            nmembers = self.nmembers
        
        coords = {'latitude':lat,'longitude':lon,'step':step,'member':slice(1,nmembers)}
        dadata,hsdadata,nodata,truthdata = self.select(variable,coords,ensmean = False)
        
        fig,ax = plt.subplots(figsize = (10,5))
        
        data = {'All Vars DA':dadata.isel(member = 0),'No DA':nodata.isel(member = 0),\
                'Hs DA':hsdadata.isel(member = 0)}
        styles = {'All Vars DA':'g','Hs DA':'b','No DA':'r','Truth':'k'}
        with plt.rc_context({'lines.linewidth':0.75}):
            self.timeseries(data,styles,'','',ax = ax)
            for i in range(1,nmembers):
                data = {'All Vars DA':dadata.isel(member = i),'No DA':nodata.isel(member = i),\
                        'Hs DA':hsdadata.isel(member = i)}
                self.timeseries(data,styles,'','',ax = ax,withlabels = False)
        
        self.timeseries({'Truth':truthdata},styles,'Leadtime (days)',\
                        self.varnames[variable]+self.units[variable],
                        ax = ax)
        # ax.set_title('Variable: ' + variable + ', Position: ({:},{:})'.format(lat,lon))
        ax.legend()
    
    def rmsebyday(self,step,lat,lon,variable):
        
        coords = {'latitude':lat,'longitude':lon}
        dadata,hsdadata,nodata,truthdata = self.select(variable,coords)
        
        dadiff = dadata - truthdata
        hsdadiff = hsdadata - truthdata
        nodiff = nodata - truthdata
        
        # find available leadtimes
        lts = np.arange(max(0,step - np.max(dadiff.step)),min(7,step))
        
        darmse = [np.sqrt((dadiff.sel(step = step - lt,leadtime = slice(lt,lt+1))**2).mean())\
                  for lt in lts]
        hsdarmse = [np.sqrt((hsdadiff.sel(step = step - lt,leadtime = slice(lt,lt+1))**2).mean())\
                      for lt in lts]
        normse = [np.sqrt((nodiff.sel(step = step - lt,leadtime = slice(lt,lt+1))**2).mean())\
                  for lt in lts]
        
        data = {'All Vars DA':darmse,'Hs DA':hsdarmse,'No DA':normse}
        styles = {'All vars DA':'k:','Hs DA':'k--','No DA':'k'}
        
        self.timeseries(data,styles,'Leadtime (days)',x = lts+0.5,figsize = (9,6),\
                        ylabel = self.varnames[variable]+self.units[variable])
        
        return darmse,normse
    
    def rmseallruns(self,lat,lon,variable,window = 12):
        
        coords = {'step':slice(1,None),'latitude':lat,'longitude':lon}
        
        dadata = self.dadata[variable].sel(coords).mean(dim = 'member')
        nodata = self.nodata[variable].sel(coords).mean(dim = 'member')
        truthdata = self.truthdata[variable].sel(coords)
        
        dadiff = dadata - truthdata
        nodiff = nodata - truthdata
        
        darmse = np.sqrt((dadiff**2).rolling(leadtime = window)\
                         .mean().mean(dim = 'step').dropna('leadtime'))
        normse = np.sqrt((nodiff**2).rolling(leadtime = window)
                         .mean().mean(dim = 'step').dropna('leadtime'))
        
        fig,ax = plt.subplots(figsize = (10,5))
        darmse.plot.line('k:',label = 'DA',ax = ax)
        normse.plot.line('k',label = 'No DA',ax = ax)
        ax.legend()
        
        return darmse,normse
    
    def peakerrors(self,step,lat,lon,variable,lt):
        
        coords = {'step':step-lt,'latitude':lat,'longitude':lon,\
                  'leadtime':slice(lt,lt+1)}
        dadata,hsdadata,nodata,truthdata = self.select(variable,coords,ensstat = None)
        
        truthi = np.argmax(truthdata.data)
        truthmag = truthdata.data[truthi]
        
        daindices = np.array([np.argmax(dadata.isel(member = i).data)\
                             for i in range(self.nmembers)]) - truthi
        hsdaindices = np.array([np.argmax(hsdadata.isel(member = i).data)\
                                for i in range(self.nmembers)]) - truthi
        noindices = np.array([np.argmax(nodata.isel(member = i).data)\
                              for i in range(self.nmembers)]) - truthi
        
        damag = np.array([dadata.isel(member = i).data[index+truthi]\
                          for i,index in enumerate(daindices)]) - truthmag
        hsdamag = np.array([hsdadata.isel(member = i).data[index+truthi]\
                            for i,index in enumerate(hsdaindices)]) - truthmag
        nomag = np.array([nodata.isel(member = i).data[index+truthi]\
                          for i,index in enumerate(noindices)]) - truthmag
        
        fig,ax =plt.subplots()
        ax.scatter(daindices,damag,marker = 'x',color = 'g',label = 'All Vars DA')
        ax.scatter(hsdaindices,hsdamag,marker = 'x',color = 'b',label = 'Hs DA')
        ax.scatter(noindices,nomag,marker = 'x',color = 'r',label = 'No DA')
        with plt.rc_context({'lines.markersize':10}):
            ax.scatter(0,0,marker = 'x',color = 'k',label = 'Truth')
        ax.legend()
        datestring = '{:02d}/03/2022T12:00:00'.format(step+6)
        ax.set_title(f'Position: ({lat},{lon}), Variable: '+variable+\
                     ', 24hr period starting '+datestring)
        
        rmses = {'datiming':rms(daindices),'damagnitude':rms(damag),\
                 'hsdatiming':rms(hsdaindices),'hsdamagnitude':rms(hsdamag),\
                 'notiming':rms(noindices),'nomagnitude':rms(nomag)}
        
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

    def corrmap(self,variable,step,ltstart = None,ltstop = None):
        
        dadata,nodata,truthdata = self.select(variable,{'step':step,\
                                                        'leadtime':slice(ltstart,ltstop)})
        
        dacorr = xr.corr(dadata,truthdata,dim = 'leadtime')
        nocorr = xr.corr(nodata,truthdata,dim = 'leadtime')
        
        mincorr = np.minimum(np.nanmin(dacorr.data),np.nanmin(nocorr.data))
        
        fig = plt.figure(figsize = (13,5))
        dax = fig.add_subplot(1,13,(1,6),projection = ccrs.PlateCarree())
        nax = fig.add_subplot(1,13,(7,12),projection = ccrs.PlateCarree())
        cax = fig.add_subplot(1,13,13)
        damesh = dax.pcolormesh(dacorr.longitude,dacorr.latitude,dacorr.data,\
                       vmin = mincorr,vmax = 1,cmap = 'Blues')
        nomesh = nax.pcolormesh(nocorr.longitude,nocorr.latitude,nocorr.data,\
                       vmin = mincorr,vmax = 1,cmap = 'Blues')
        cbar = plt.colorbar(damesh,cax)
        cbar.set_label('Correlation')
        dax.coastlines()
        nax.coastlines()

    def rmsemap(self,step,variable,ltstart = None,ltstop = None):
        
        coords = {'step':step,'leadtime':slice(ltstart,ltstop)}
        
        dadata,hsdadata,nodata,truthdata = self.select(variable,coords)
        
        dadiff = dadata - truthdata
        hsdadiff = hsdadata - truthdata
        nodiff = nodata - truthdata
        
        darmse = np.sqrt((dadiff*dadiff).mean(dim = 'leadtime'))
        hsdarmse = np.sqrt((hsdadiff*hsdadiff).mean(dim = 'leadtime'))
        normse = np.sqrt((nodiff*nodiff).mean(dim = 'leadtime'))
        allreduction = normse - darmse
        hsreduction = normse - hsdarmse
        
        fig = plt.figure(figsize = (19,5))
        dax = fig.add_subplot(131,projection = ccrs.PlateCarree())
        nax = fig.add_subplot(132,projection = ccrs.PlateCarree())
        hax = fig.add_subplot(133,projection = ccrs.PlateCarree())
        damesh = dax.pcolormesh(darmse.longitude,darmse.latitude,allreduction.data,\
                       norm = colors.CenteredNorm(),cmap = 'RdBu')
        nomesh = nax.pcolormesh(normse.longitude,normse.latitude,normse.data,\
                       cmap = 'Reds')
        hsmesh = hax.pcolormesh(hsdarmse.longitude,hsdarmse.latitude,hsreduction.data,\
                       norm = colors.CenteredNorm(),cmap = 'RdBu')
        dacbar = plt.colorbar(damesh)
        dacbar.set_label('RMS error reduction'+self.units[variable])
        nocbar = plt.colorbar(nomesh)
        nocbar.set_label('RMS error'+self.units[variable])
        hscbar = plt.colorbar(hsmesh)
        hscbar.set_label('RMS error reduction'+self.units[variable])
        dax.coastlines()
        nax.coastlines()
        hax.coastlines()
        fig.suptitle('RMS error in '+variable+f' ensemble mean (day {ltstart} to day {ltstop})')
        dax.set_title('All Vars DA')
        nax.set_title('No DA')
        hax.set_title('Hs DA')
        fig.tight_layout()
    
    def spreadseries(self,step,lat,lon,variable):
        
        coords = {'step':step,'latitude':lat,'longitude':lon}
        
        daspread,hsdaspread,nospread,_ = self.select(variable,coords,ensstat = xr.DataArray.std)
        
        data = {'All Vars DA':daspread,'Hs DA':hsdaspread,'No DA':nospread}
        styles = {'All Vars DA':'g','Hs DA':'b','No DA':'r'}
        self.timeseries(data,styles,'Leadtime (days)',\
                        'Ensemble Standard Deviation'+self.units[variable],\
                        'Ensemble Spread in '+self.varnames[variable])
        
        return daspread,hsdaspread,nospread
    
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