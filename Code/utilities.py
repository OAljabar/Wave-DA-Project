# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:18:01 2024

@author: owena
"""



import matplotlib as mpl

def plotsettings(settings_dict):

    mpl.rc
    
    if 'linewidth' in settings_dict:
        mpl.rcParams['lines.linewidth'] = settings_dict['linewidth']
    
    mpl.rcParams['font.weight'] = 'bold'
    
    mpl.rcParams['axes.labelweight'] = 'bold'
    
    if 'linewidth' in settings_dict:
        mpl.rcParams['font.size'] = settings_dict['fontsize']