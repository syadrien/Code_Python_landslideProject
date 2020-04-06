# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:42:53 2019

@author: Adrien 


Rainfall - runoff model for Rempart Discharge

Inspire du modele rainfall-runoff de Cilaos (rainfall_runoff_cilaos.py)
La majorite du code est le meme a l'exception de : 
    -import different dataset
    -calibration de certains paramatere different
Les parametres ont ete "pre-calibre" avec le modele tableau modelRRrempart.xls 
Etant donne les donnes tres limitee pour riviere Rempart, on reutilise la calibration de Cilaos

"""



import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import tasgrid as tg
import scipy
#import utm
import peakutils
from peakutils.plot import plot as pplot
import math
import statsmodels.formula.api as sm
from matplotlib import dates as mdates

from datetime import datetime

import mpld3

from scipy.signal import find_peaks


#%% Import data


os.chdir("C:/Users/Reika/Desktop/pluieRempart")




#Import pluie
pluie7780=pd.read_csv('C:/Users/Reika/Desktop/pluieRempart/pluvio1998_2008/compilPluieRempart.csv', sep=',',encoding="ISO-8859-1")

#pluie7780=pd.read_csv('C:/Users/Reika/Desktop/pluieRempart/pluieCilaosDaily.csv', sep=',',encoding="ISO-8859-1")

pluie7780['DATE'] = pd.to_datetime(pluie7780['DATE'],format='%m/%d/%Y')
pluie7780=pluie7780.set_index(pd.DatetimeIndex(pluie7780['DATE']))
pluie7780=pluie7780.resample('D').mean()
pluie7780['month']=pluie7780.index.month
pluie7780['year']=pluie7780.index.year

#pluie7780=pluie7780.dropna()


# Import donnee debit
debit = pd.read_csv("C:/Users/Reika/Dropbox/PhD_Doc/claassification-hydro/DonneDebitMoyenJournalier/export_donnees_journalieres_46136.csv",sep=';',encoding='cp1252',header=7)
debit['DATE'] = pd.to_datetime(debit['DATE'],format='%Y-%m-%d')
debit=debit.set_index(pd.DatetimeIndex(debit['DATE']))
debit['month']=debit.index.month
debit['year']=debit.index.year

#merge=pluie7780
merge=pd.merge(pluie7780, debit, left_index=True, right_index=True,how='left')
merge['DebitMinBis']=merge.Dischagre_L_Sec.rolling(min_periods=1,window=30).min()
merge['DebitMin']=merge['DebitMinBis'][~np.isnan(merge['Dischagre_L_Sec'])]

figHydrogram=plt.figure()        
ax1 = figHydrogram.add_subplot(211)
line1=ax1.bar(merge.index.values,merge.RR,label='pluie')
#line2=ax1.bar(merge[merge.RR.isna()].index.values,100*np.ones(np.size(merge[merge.RR.isna()].index.values)),color='black',alpha=0.5,label='No Data')
#line3=ax1.plot(merge.DATE,merge['RR'],'b1',label='Pluie')
ax1.set_ylabel('pluie in mm/day', color='b')
ax1.xaxis_date()
plt.legend(loc=2)
ax2 = figHydrogram.add_subplot(212, sharex=ax1) 
line3=ax2.plot(debit.Dischagre_L_Sec,'go',label="Debit Brut in L/s",markersize=5)
#line2=ax2.plot(debit.DebitMin,'r.',label="Debit Base in L/s")
#line4=ax2.plot(debit.DebitPeak,'b*',label="Debit peak in L/s")
ax2.set_ylabel('debit in l/s', color='r')
#ax2.yaxis.set_label_position("right")
ax2.xaxis_date()
plt.show()
plt.legend()

#merge.groupby(merge.index.year).count();

#%%


cu=40000
su0=18000
td=1
qsu0=200
ru=450
si_scale=1
si0=10
Qof_scaling=1

ss0=200000
m=70000
qs0=200

#merge['DIFFERENCE_DEBIT']=(merge.DebitMin-merge.Dischagre_L_Sec)


#apres optimisation des paramatres ru et m : 
ru = 280
m = 60000



merge['sux']=np.nan
merge['su']=np.nan
merge['si']=np.nan
merge['Qus']=np.nan
merge['Qof']=np.nan
merge['Ss']=np.nan
merge['Qs']=np.nan
merge['sumQof']=np.nan
merge['sumQpeak']=np.nan
merge['sumQs']=np.nan


merge['sux'][0]=su0+merge['RR'][0]*ru-qsu0
merge['su'][0]=merge['sux'][0]
merge['si'][0]=(cu-merge['su'][0])*si_scale
merge['Qus'][0]=merge['su'][0]/merge['si'][0]/td
merge['Qof'][0]=0
merge['Ss'][0]=ss0
merge['Qs'][0]=qs0*np.exp(merge['Ss'][0]/m)


for ru in np.arange(270,320,10):
    for m in np.arange(58000,61000,1000):


        for i in np.arange(1,len(merge)):
            
            merge.at[merge.index[i],'sux']=merge['su'][i-1]-merge['Qus'][i-1]+merge['RR'][i]*ru
            
            if merge['sux'][i]>cu:
                merge.at[merge.index[i],'su']=cu
            else :
                merge.at[merge.index[i],'su']=merge['sux'][i]
            
            if (cu-merge['su'][i])*si_scale>si0:
                merge.at[merge.index[i],'si']=(cu-merge['su'][i])*si_scale
            else : 
              merge.at[merge.index[i],'si']=si0        
              
            merge.at[merge.index[i],'Qus']=  merge['su'][i]/merge['si'][i]/td
            
            if merge['sux'][i]>cu:
                merge.at[merge.index[i],'Qof']=(merge['sux'][i]-cu)*Qof_scaling
            else :
                merge.at[merge.index[i],'Qof']=0
                
            merge.at[merge.index[i],'Ss']=merge['Ss'][i-1]-merge['Qs'][i-1]+merge['Qus'][i-1]
            
            merge.at[merge.index[i],'Qs']=qs0*np.exp(merge['Ss'][i]/m)
            
               
         
                           
        
            
        indexQof=(np.nonzero(merge['Qof']))                 
        dateQof=merge.index[indexQof]           
                           
        for val in np.nditer(indexQof) :
            
            if val<5477 : 
            
                merge.at[merge.index[int(val)],'sumQof']=merge['Qof'][val]+merge['Qof'][val+1]+merge['Qof'][val+2]+merge['Qof'][val+3]+merge['Qof'][val+4]
                
                merge.at[merge.index[int(val)],'sumQs']=merge['Qs'][val]+merge['Qs'][val+1]+merge['Qs'][val+2]+merge['Qs'][val+3]+merge['Qs'][val+4]
            
        
        # in a windows, we dont want multiple/succesive values, remove other value 
        for i in np.arange(1,len(merge)-4):
            if np.isnan(merge['sumQof'][i])==False:
                merge.at[merge.index[i+1],'sumQof']=np.nan                      
                merge.at[merge.index[i+2],'sumQof']=np.nan                      
                merge.at[merge.index[i+3],'sumQof']=np.nan                      
                merge.at[merge.index[i+4],'sumQof']=np.nan                      
        
                
                
        #peaks, _ = find_peaks(merge['sumQof'], height=0,distance=1)
        #
        #merge['filtersumQof']=merge['sumQof'][peaks]
        
             
        # Peak flow defined as diff measured - Qs, check corr de sum
        merge['MeasuredPeakQ']=merge['Dischagre_L_Sec']-merge['Qs']
        merge['sumMeasuredPeakQ']=np.nan
        
        for i in np.arange(1,len(merge)-4):
            merge.at[merge.index[i],'sumMeasuredPeakQ']=merge['MeasuredPeakQ'][i]+merge['MeasuredPeakQ'][i+1]+merge['MeasuredPeakQ'][i+2]+merge['MeasuredPeakQ'][i+3]+merge['MeasuredPeakQ'][i+4]
        
                       
        for val in np.nditer(indexQof) :
            
            if val<5477 : 
        
                merge.at[merge.index[int(val)],'Qs']=np.nan
                merge.at[merge.index[int(val+1)],'Qs']=np.nan
                merge.at[merge.index[int(val+2)],'Qs']=np.nan
                merge.at[merge.index[int(val+3)],'Qs']=np.nan
                merge.at[merge.index[int(val+4)],'Qs']=np.nan
                            
                            
                            
        for k in np.arange(1,len(merge)):
            merge.at[merge.index[int(k)],'sumQpeak'] =0.3327*merge['sumQof'][k]+merge['sumQs'][k]         
        
        
        
        
        
        
        merge['mergeModelQ'] = merge['Qs'].fillna(merge['sumQpeak'])
        
        
        
        # create same array with measure discharge
        merge['MeasuredSumQ'] = merge['Dischagre_L_Sec']
        
        for val in np.nditer(indexQof) :
            if val<5477 : 
                merge.at[merge.index[int(val)],'MeasuredSumQ']=np.nan
                merge.at[merge.index[int(val+1)],'MeasuredSumQ']=np.nan
                merge.at[merge.index[int(val+2)],'MeasuredSumQ']=np.nan
                merge.at[merge.index[int(val+3)],'MeasuredSumQ']=np.nan
                merge.at[merge.index[int(val+4)],'MeasuredSumQ']=np.nan
        
        
        for val in np.nditer(indexQof) :
            if val<5477 : 
            
                merge.at[merge.index[int(val)],'MeasuredSumQ']=merge['Dischagre_L_Sec'][val]+merge['Dischagre_L_Sec'][val+1]+merge['Dischagre_L_Sec'][val+2]+merge['Dischagre_L_Sec'][val+3]+merge['Dischagre_L_Sec'][val+4]
            
          
            
        #merge['mergeModelQ'] = merge['Qs'].fillna(merge['sumQpeak'])
           
            
        
        
        
        
        
        
        # Hydrogram avec tout les debit merged
        
        figHydrogram=plt.figure()        
        ax1 = figHydrogram.add_subplot(211)
        line1=ax1.bar(merge.index.values,merge.RR,label='pluie')
        #line3=ax1.plot(merge.DATE,merge['RR'],'b1',label='Pluie')
        ax1.set_ylabel('pluie in mm/day', color='b')
        ax1.xaxis_date()
        plt.legend(loc=2)
        ax2 = figHydrogram.add_subplot(212, sharex=ax1) 
        line3=ax2.plot(merge.index.values,merge.mergeModelQ,'ro',label='Modelled discharge')
        line2=ax2.plot(merge.DATE,merge.MeasuredSumQ,'g*',label='Measured Discharge')
        #ax2.yaxis.tick_right()
        ax2.set_ylabel('debit in l/s', color='r')
        #ax2.yaxis.set_label_position("right")
        ax2.xaxis_date()
        plt.show()
        plt.legend()
        
        
        
        merge['Ecart-model']=abs(merge.mergeModelQ-merge.MeasuredSumQ)
        merge['DIFF_proportion']=(merge['Ecart-model']/merge.MeasuredSumQ*100) 
        print('ru = '+str(ru))
        print('m = '+str(m))
        print('ecart moyen = ' + str(np.nanmean((merge['DIFF_proportion'])))+'%')
        print('---------------------')


merge['Q_m3/day']=merge.mergeModelQ*86.4
A=merge.groupby(['year_x']).sum()['Q_m3/day']


#%%

# Hydrogram avec disinction des debits modelled

figHydrogram=plt.figure()        
ax1 = figHydrogram.add_subplot(211)
line1=ax1.bar(merge.index.values,merge.RR,label='pluie')
ax1.set_ylabel('pluie in mm/day', color='b')
ax1.xaxis_date()
plt.legend(loc=2)
ax2 = figHydrogram.add_subplot(212, sharex=ax1) 
line2=ax2.plot(merge.index.values,merge.Qs,'b.',label='Qs')
line3=ax2.plot(merge.index.values,merge.sumQpeak,'g*',label='Sum Peak')
line4=ax2.plot(merge.DATE,merge.MeasuredSumQ,'ro',label='Measured Discharge')
#ax2.yaxis.tick_right()
ax2.set_ylabel('debit in l/s', color='r')
#ax2.yaxis.set_label_position("right")
ax2.xaxis_date()
plt.show()
plt.legend()


#%%  Correlation plot 
plt.figure()
plt.plot(merge['DebitMin'],merge['Qs'],'k.',label='Data')  
plt.plot(np.arange(12000),np.arange(12000),'g-',label='Ideal fit')  
plt.xlabel('Debit min in L/sec')
plt.ylabel('Qs in L/sec')
plt.title('Baseflow correlation plot ¦ Correlation=' + str(round(merge['DebitMin'].corr(merge['Qs']), 3)))
plt.legend()


plt.figure()
plt.plot(merge['sumMeasuredPeakQ'],merge['sumQpeak'],'k.',label='Data')  
plt.plot(np.arange(50000),np.arange(50000),'g-',label='Ideal fit')  
plt.xlabel('Cumulative measured peak discharge')
plt.ylabel('Cumulative Qpeak')
plt.title('Peak flow correlation plot ¦ Correlation=' + str(round(merge['sumMeasuredPeakQ'].corr(merge['sumQpeak']), 3)))
plt.legend()     
    
    
plt.figure()
plt.plot(merge['MeasuredSumQ'],merge['mergeModelQ'],'k.',label='Data')  
plt.plot(np.arange(5200),np.arange(5200),'g-',label='Ideal fit')  
plt.xlabel('Measured discharge in L/sec')
plt.ylabel('Modelled discharge in L/sec')
plt.title('Discharge correlation plot ¦ Correlation=' + str(round(merge['MeasuredSumQ'].corr(merge['mergeModelQ']), 2)))
plt.legend()




#%%  Une fois qu'on a le debit modelise pour Cilaos, distinction entre base and peakflow pour ce BV

# ici Qs = baseflow   et Qof = debit peak, pas besoin de distinction avec rolling min

# in l/year     Z
DebitBase=merge.groupby(['year_x']).sum()['Qs']*86400

DebitPeak=merge.groupby(['year_x']).sum()['sumQpeak']*86400
