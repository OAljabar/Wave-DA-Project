#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:00:52 2024

@author: mg838076
"""

from outputstats import outputdata,outputstats
import os

scaledir = os.environ.get('SCALEDIR')
constdir = os.environ.get('CONSTDIR')

data = outputdata(scaledir)
data.readall()
stats_scale = outputstats(data)

data = outputdata(constdir)
data.readall()
stats_const = outputstats(data)

lat = 18.4
lons = [-93.7,-85]
variables = ['hs','t01']
steps = [1,2,3,4,5]

paramslist = [[lat,lon,var,step]\
          for lon in lons for var in variables for step in steps]

filenames = [f'lon{lon}'+var+f'step{step}.png' for [lat,lon,var,step] in paramslist]

for i,params in enumerate(paramslist):
    stats_const.rmse(params[0],params[1],params[2],params[3],savefig = True,\
                     filename = '/home/users/mg838076/Dissertation/figures/constfigs/'+filenames[i])
    stats_scale.rmse(params[0],params[1],params[2],params[3],savefig = True,\
                     filename = '/home/users/mg838076/Dissertation/figures/scalefigs/'+filenames[i])
        