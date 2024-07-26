# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:04:58 2024

@author: owena
"""

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.stats as stats

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
    artist = ax.hist(data,bins)
    
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
    
    #return artist

def histanim(ens):
    
    times = ens.data.time.thin(6).data
    artists = []
    for time in times:
        artists.append(enshist(ens,[20,-95],time,0.5,6))
    
    return artists