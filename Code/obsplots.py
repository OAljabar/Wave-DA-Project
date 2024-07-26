# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:42:51 2024

@author: owena
"""

from obsdata import obsplots
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

filepath = '..\\..\\data\\Buoy_Spectral_20210903_20220413\\parameters.txt'

def freqbands():
    obs = obsplots(filepath,variables = ['TM02A','TM02B','TM02C','TP','TPC','HM0A','HM0B','HM0C'])
    
    fig,axs = plt.subplots(nrows = 3,figsize = (18,15))
    obs.plotvars(['TM02A','TM02B','TM02C'],'s','Period',axs[0])
    obs.plotvars(['TP','TPC'],'s','Peak period',axs[1])
    obs.plotvars(['HM0A','HM0B','HM0C'],'m','Wave height',axs[2])

def dirplots(time):
    variables = ['HM0','MDIR']#'HM0A','HM0B','HM0C','MDIRA','MDIRB','MDIRC']
    
    obs = obsplots(filepath,variables)
    obs.data = obs.data.sel(time = time)
    times = (obs.data.time - obs.data.time[0]).astype('float')/3.6e12
    fig = plt.figure(figsize = (7,7))
    ax = fig.add_subplot(projection = 'polar',theta_offset = np.pi/2,theta_direction = -1)
    ax.plot(np.deg2rad(obs.data['MDIR']),obs.data['HM0'],'k:')
    crosses = ax.scatter(np.deg2rad(obs.data['MDIR']),obs.data['HM0'],marker = 'x',c = times)
    plt.colorbar(crosses)
    ax.set_xticks([0,np.pi/2,np.pi,3*np.pi/2],['N','E','S','W'])

def dirstd():
    obs = obsplots(filepath,variables = ['HM0','MDIR'])
    magnitude,std = obs.std(nbins = 20,evenbins = False,reldif = False,\
                            xvar = 'MDIR',yvar = 'HM0',fitline = False)
    
    fig,ax = plt.subplots(subplot_kw={'projection':'polar','theta_offset':np.pi/2,\
                                      'theta_direction':-1})
    ax.plot(np.deg2rad(magnitude),std)
    ax.set_xticks([0,np.pi/2,np.pi,3*np.pi/2],['N','E','S','W'])

def diffdirscatter(run = 6,nmembers = 30,finedomain = False):
    
    obs = obsplots(filepath,variables = ['HM0','MDIR'])
    obs.getensdata(run,nmembers,True)
    diff,ensmean = obs.ensdiff()
    obs.data = obs.data.sel(time = slice(min(diff.time),max(diff.time)))
    
    fig,ax = plt.subplots(subplot_kw={'projection':'polar','theta_offset':np.pi/2,\
                                      'theta_direction':-1})
    ax.scatter(np.deg2rad(obs.data['MDIR']),diff,marker = 'x')
    ax.set_xticks([0,np.pi/2,np.pi,3*np.pi/2],['N','E','S','W'])
    
def hdirscatter(obs = None):
    
    if obs is None:
        obs = obsplots(filepath,variables = ['HM0','MDIR'])

    fig = plt.figure(figsize = (7,5))
    ax = fig.add_subplot(projection = 'polar',theta_offset = np.pi/2,theta_direction = -1)
    _,_,_,dplot = ax.hist2d(np.deg2rad(obs.data['MDIR']),obs.data['HM0'],bins = 40,cmap = 'Blues',\
              norm = colors.LogNorm())
    plt.colorbar(dplot,ax = ax)
    ax.set_xticks([0,np.pi/2,np.pi,3*np.pi/2],['N','E','S','W'])