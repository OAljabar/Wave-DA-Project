# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:49:14 2024

@author: owena
"""

from model import FugroModel
import matplotlib.pyplot as plt

filedir = '..\\..\\data'

model = FugroModel(6,10,filedir = filedir)

model.init_fields();
model.state['hs'].plot()
plt.savefig('..\\..\\frames\\0.png')
plt.clf()

for i in range(48):
    for j in range(2):
        model.step(model.time);
    plt.figure()
    model.state['hs'].plot()
    print(model.time)
    plt.savefig('..\\..\\frames\\{:}.png'.format(model.time))
    plt.clf()