# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:52:36 2024

@author: owena
"""

from outputstats import outputdata,outputstats

# directory containing output files
outputdir = ''

data = outputdata(outputdir)
data.readall() # can't use input() to get file format

stats = outputstats(outputdir)
