# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:45:24 2019

@author: Adrien

Temporal interpolation of evapotranspiration data + calcul of inter-annual mean 
Data for evapotranspiration are temporally scarce, so they are being temporally interpolated using a polynomialof order 4

Based on classPET.py
Update avec valeurs de PET correcte


"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#%%

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

listMindiffCourante=[]
listMaxdiffCourante=[]
listMeandiffCourante=[]
listMinCVCourant=[]
listMaxCVCourant=[]
listMeanCVCourant=[]

listRMSE=[]


os.chdir("C:/Users/Reika/Desktop/GOOD/data_PET-12_4/")

dataRaw=pd.read_csv('dataETPpartialClean.csv', sep=';')
dataRaw.DATE=pd.DatetimeIndex(dataRaw.DATE)
dataRawbis=dataRaw.set_index('STATION')

infoStation=pd.read_csv('infoStation.csv', sep=',')




infoStation['Moyenne'] = np.nan
infoStation['STD'] = np.nan
infoStation['CVdiff'] = np.nan


infoStation=infoStation.set_index('Numero')

RapportSTDDebitMoyen=[]

#count of number of years with available data
NbreAn=0
sumNbreAn=0

# count ot number of available data
NbreDonneETP=[]

for i in np.unique(dataRawbis.index) :
     
# keep track of station ID during loop                                                                              
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')          
    print(i)
    print('-------------------------')    
    
    TabDiff=[]
    
    diffMois=np.zeros(12)
    
    donneeCourante=dataRawbis.loc[i]
    NbreDonneETP.append(len(donneeCourante))
    donneeCourante=donneeCourante.set_index('DATE')        


    index=pd.date_range(str(min(donneeCourante.index)), str(max(donneeCourante.index)),freq='MS')
    donneeCourante = donneeCourante.reindex(index)
        
    donneeCouranteBis=donneeCourante[np.isfinite(donneeCourante.ETP)]
    
    
    
    STDmonth=[]
    NbreMois=[]
    
    donneeCouranteBis['month']=donneeCouranteBis.index.month
    donneeCourante['month']=donneeCourante.index.month

    NbreData=len(donneeCouranteBis) 
    
    
    #INTERPOLATION USING A 4TH ORDER POLYNOMIAL    
    def func(x, a, b, c, d,e):
        return a*x**4+b *x**3+c*x**2+d*x+e

        
    Y=donneeCouranteBis.ETP.values
    X=donneeCouranteBis.index
    
   
    popt, pcov = curve_fit(func, X.month, Y)

    
    # have the same index as x ie donneeCouranteBis
    diffCourante=Y-func(X.month, *popt)
    
    
    # calculation of r2 of fit
    residuals = Y-func(X.month, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Y-np.mean(Y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    STDest=np.sqrt(ss_res/len(X))
    
     
    
    meandiffCourante=np.nanmean(abs(diffCourante))
    
    donneeCouranteBis['diff']=donneeCouranteBis.ETP.values-func(X.month, *popt)

# calculation ofthe error for each month of the year based on diffCourante + index of X
    
    monthlyDiff=donneeCouranteBis['diff'].groupby(donneeCouranteBis.month).mean()
    NbreMois=donneeCouranteBis['diff'].groupby(donneeCouranteBis.month).count()
    
#    array with monthly diff in average
    MonthDiff=np.empty((12,1,))
    MonthDiff[:] = np.nan
    
    
    # shift of the index by 1
    for mois in monthlyDiff.index.values-1:
        MonthDiff[mois]=monthlyDiff[mois+1]
        
  
    equation =str(round(popt[0],2))+'*x^4+'+str(round(popt[1],2))+'*x^3+'+str(round(popt[2]))+'*x^2+'+str(round(popt[3]))+'+*x+'+str(round(popt[4]))
  
#    plot to see timeserie when we have available data
    plt.figure()
    plt.plot(donneeCourante.ETP, 'b*', label='Raw Data',ms=10)
    plt.plot(donneeCourante.index, func(donneeCourante.month, *popt), 'rs', label='Interpolated data \nEquation ',ms=10)
    plt.title('Temporal evolution of monthly PET for raw and interpolated values for stations '+str(i)+'\nEquation'+ equation)
    plt.xlabel('Year',fontsize=18)
    plt.ylabel('Monthly PET in mm/month',fontsize=18)
    plt.legend()

    
  
    plt.figure()
    sns.pointplot(data=donneeCourante,x=donneeCourante.month.values,y=donneeCourante.ETP.values,hue=donneeCourante.index.year,linestyles='--',zorder=100,marker='.') 
    plt.errorbar(np.arange(0,12), func(np.arange(1,13), *popt),yerr=MonthDiff, lw=5,c='black',zorder=1,marker='s',markersize=20)
#    plt.plot(np.arange(0,12), func(np.arange(1,13), *popt), lw=5,c='black',zorder=100,marker='s',markersize=20)
    plt.ylabel('Monthly PET in mm/month')
    plt.xlabel('Month')
    plt.title('Annual variation of monthly PET and interpolated values for station '+str(i) + '\n R2 = '+str(round(r_squared,2)))
    plt.xticks(np.arange(0,13),['J','F','M','A','M','J','J','A','S','O','N','D'])
    plt.show()
    
    
    for xpos, ypos, name in zip(np.arange(0,12), func(np.arange(0,12), *popt), NbreMois):
        plt.annotate(name, (xpos, ypos), xytext=(0, 110), va='bottom',
                    textcoords='offset points',fontsize=14,zorder=10)
    
    
# estimatiom of fit error
# one way is to estimate the mean error for each month when there are data, comparing the polynomial and real (measured) values
# this give us the mean error over a month, which is then mutlitplied by 12 because for some stations, some month of the year do not have data at all
    
    donneeCouranteInterp=func(np.arange(1,13), *popt)

    
    sumAnnuel=np.nansum(donneeCouranteInterp)
    STDAnnuel=np.nanmean(donneeCouranteInterp)

    infoStation.loc[i,'STDest']=STDest*12
    infoStation.loc[i,'STDestin%']=STDest*12/np.mean(sumAnnuel)*100
    infoStation.loc[i,'R2']=r_squared
    infoStation.loc[i,'SumMonth']=np.mean(sumAnnuel)
    infoStation.loc[i,'STDMonth']=np.mean(STDAnnuel)
    infoStation.loc[i,'MeanDiff']=meandiffCourante*12
    infoStation.loc[i,'MeanDiffin%']=meandiffCourante*12/np.mean(sumAnnuel)*100
    infoStation.loc[i,'NbreData']=NbreData