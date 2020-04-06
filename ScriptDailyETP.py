#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 12:41:37 2018

@author: adrien



Classification donnee ETP Dailt Gillot Airport


"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tasgrid as tg
import utm



#%%    Plot evo temporelle ETP

plt.close('all')

dataRaw=pd.read_csv('GillotDaily.csv', sep=';')
dataRaw.DATE=pd.DatetimeIndex(dataRaw.DATE)
dataRawbis=dataRaw.set_index('DATE')

plt.figure()
plt.plot(dataRawbis.ETPMON,'b*')

Avg=np.nanmean(dataRawbis.ETPMON)
Var=np.nanvar(dataRawbis.ETPMON)
Std=np.nanstd(dataRawbis.ETPMON)

Rapport=Std/Avg*100