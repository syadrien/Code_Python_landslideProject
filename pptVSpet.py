# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:03:28 2019

@author: Adrien


Comparaison PET and PPT record for identic station in time

Test whether PET = RET or if enough rain to evaporate
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#%%


os.chdir("C:/Users/Reika/Dropbox/PhD_Doc/classification-ETP/")

dataRaw=pd.read_csv('dataETPpartialClean.csv', sep=';')
dataRaw.DATE=pd.DatetimeIndex(dataRaw.DATE)
dataRawbis=dataRaw.set_index('STATION')

PET=dataRawbis.loc[97422440]
PET.index = pd.to_datetime(PET.DATE)


PPT=pd.read_csv('C:/Users/Reika/Dropbox/PhD_Doc/Precipitation/pluvio1998_2008/Pluvio-Quot-PLaineDesCafres-9808.txt', sep=',')

PPT['DATE'] = pd.to_datetime(PPT['DATE'])
PPT.index = pd.to_datetime(PPT.DATE)
PPT=PPT.drop(labels='DATE',axis=1)


#pluie=PPT.groupby([PPT.index.year.rename('year'), PPT.index.month.rename('month')]).sum()
#pluie['month']=pluie.index.month

a,b=0,2000

A=PPT.resample('MS').sum()

Reff=A.RR-PET.ETP

###
fig1 = plt.figure()

#plt.title('Rainfall, PET and Reff for station 97422440')
ax1 = fig1.add_subplot(211)
plt.title('Monthly cumulative rainfall, PET and Reff for station 97421210',fontsize=20)
ax1.bar(A.index.values-np.timedelta64(5,'D'),A.RR,color='blue',width=10,label='Monthly rainfall in mm/month')
ax1.xaxis_date()
#plt.plot(A.index,A.RR,'g.-',label='Monthly rainfall in mm/month')
plt.ylabel("Monthly rainfall in mm/month",color='b',fontsize=15)
ax1.tick_params(axis='y', colors='b',size=15)
ax1.set_ylim(a,b)
ax2 = fig1.add_subplot(211, sharex=ax1, frameon=False)
ax2.bar(PET.index.values + np.timedelta64(5,'D'),PET.ETP,color='red',width=11,label='Monthly PET in mm/month')
ax2.xaxis_date()
#plt.plot(PET.index,PET.ETP,'r.-',label='Monthly PET in mm/month')
ax2.yaxis.tick_right()
ax2.tick_params(axis='y', colors='r',size=15)
ax2.yaxis.set_label_position("right")
plt.ylabel("Monthly PET in mm/month",color='r',fontsize=15)
ax2.set_ylim(a,b)
#plt.gca().invert_yaxis()
plt.show()

ax3 = fig1.add_subplot(212, sharex=ax1)
plt.plot(Reff.index,Reff.values,'k.-',label='Monthly Effective Rainfall in mm/month')
plt.ylabel("Monthly Effective Rainfall in mm/month",color='k',fontsize=15)
ax3.tick_params(axis='y', colors='k',size=15)
ax3.axhline(y=1,color='k',linestyle='--')

#plt.grid()


# si different time period, on prend moyenne mensuelle sur plusieurs annees

AnPET=PET.groupby(PET.index.month).mean()
AnPluie=A.groupby(A.index.month).mean()
AnReff=AnPluie.RR-AnPET.ETP
#


fig2=plt.figure()
#plt.title('Rainfall, PET and Reff for station 97422440')
ax1 = fig2.add_subplot(211)
plt.title('Rainfall, PET and Reff for station 97421210',fontsize=20)
plt.plot(AnPluie,'g.-',label='Monthly rainfall in mm/month')
plt.ylabel("Monthly rainfall in mm/month",color='g',fontsize=15)
plt.xlabel("Month",fontsize=15)
ax1.tick_params(axis='y', colors='g',size=15)
ax1.set_ylim(a,b)
ax2 = fig2.add_subplot(211, sharex=ax1, frameon=False)
plt.plot(AnPET.ETP,'r.-',label='Monthly PET in mm/month')
ax2.yaxis.tick_right()
ax2.tick_params(axis='y', colors='r',size=15)
ax2.yaxis.set_label_position("right")
plt.ylabel("Monthly PET in mm/month",color='r',fontsize=15)
ax2.set_ylim(a,b)
#plt.gca().invert_yaxis()
plt.show()

ax3 = fig2.add_subplot(212, sharex=ax1)
plt.plot(AnReff,'b.-',label='Monthly Effective Rainfall in mm/month')
plt.ylabel("Monthly Effective Rainfall in mm/month",color='b',fontsize=15)
plt.xlabel("Month",fontsize=15)
ax3.tick_params(axis='y', colors='b',size=15)
plt.grid()

