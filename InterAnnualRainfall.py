# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:08:22 2019

@author: Adrien SY

Calculation for inter-annual rainfall for several station, between years 1998 and 2008

Input : Several tables Yi of daily rainfall for several station, stored in a folder
        A table X containing information about all station on Reunion Island
        
Goal : for each of the station for which we have rainfall data (ie table Yi), add to the table X a new column with inter-annual rainfall          

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn


#%% Calculation of inter-annual rainfall by summing the inter annual monthly rainfall for each months (J to D)
# This is method is used because some data are misssing and we want to ensure that missing data are not biasing the final results
# We only used the complete months (months for which we have more than 28 days of data)

# Monthly average
 
plt.close('all')

#open the folder where data are located
os.chdir("C:/Users/Reika/Dropbox/PhD_Doc/Precipitation/")

# open the table X with all station info
infoStation=pd.read_csv('InfoStattion_12_4.csv', sep=',',encoding="ISO-8859-1")

# create new column where we willl store the new info, ie Mean and standard deviation for years 1998to 2008
infoStation['MeanAnnualRain98-08inmm/yr']=np.nan
infoStation['STDRain98-08inmm/yr']=np.nan
 
# open the folder where the table Yi are stored
os.chdir("C:/Users/Reika/Dropbox/PhD_Doc/Precipitation/pluvio1998_2008")

#store name of files in the folder
files=os.listdir("C:/Users/Reika/Dropbox/PhD_Doc/Precipitation/pluvio1998_2008")


listStation=[]
compil=pd.Series([])

Allcompil=pd.Series([])

monthlyMean=pd.DataFrame

#browse through all the files in the folder, containing Yi tables

for i in files :
        
    Mois=[]
    MonthlyInterAnnualMean=[]
    MonthlyInterAnnualSTD=[]
    MonthlyInterAnnualCV=[]
    
    DataCurrentStation=pd.read_csv(i, sep=',',encoding="ISO-8859-1")

    
    currentStation=DataCurrentStation.POSTE[1]
    listStation.append(currentStation);    

    Tmeasure=pd.to_datetime(DataCurrentStation.DATE).dt.date.tolist();
    Measure=DataCurrentStation.RR;     
    

    CroppedData=pd.DataFrame(index=Tmeasure)                        
        
    CroppedData['Measure'] = Measure.values;
    CroppedData.index = pd.to_datetime(CroppedData.index)
    
    
    CroppedData['month']=CroppedData.index.month
    CroppedData['year']=CroppedData.index.year
    
    # mean and std for a given month
    compil_mean=[]
    compil_std=[]
    compil_var=[]

    
    # browse through each monthof the year (J to D)
    for j in np.arange(1,13):
        
        # mean and std for a given month

        MonthlyMeasure=CroppedData[CroppedData['month']==j]
        NbreMeasureDansMois=MonthlyMeasure.Measure.groupby(MonthlyMeasure.index.year).count()
        #find complete month ie month where #day>28
        years=NbreMeasureDansMois[NbreMeasureDansMois >= 28].index
        CompleteMonth=MonthlyMeasure[MonthlyMeasure['year'].isin(years)]
        

        
        mean_month_rain=np.mean(CompleteMonth.groupby([(CompleteMonth.index.year),(CompleteMonth.index.month)]).sum().Measure.values)
        std_month_rain=np.std(CompleteMonth.groupby([(CompleteMonth.index.year),(CompleteMonth.index.month)]).sum().Measure.values)
        var_month_rain=np.var(CompleteMonth.groupby([(CompleteMonth.index.year),(CompleteMonth.index.month)]).sum().Measure.values)


        # vecteur avec pour chaque mois, la moyenne/std mensuelle des pluies
        compil_mean.append(mean_month_rain)
        compil_std.append(std_month_rain)
        compil_var.append(var_month_rain)
        
        # Annual mean of monthly rainfall for each year between 98 and 08, for complete months, somme sur 12mois
        AnnualMean=np.sum(compil_mean)
        # sum = sum over 12 month..OK
        AnnualSTD=np.sqrt(np.mean(compil_var))
        # STD =  average the variances; then you can take square root to get the average standard deviation.
        


    localisation=infoStation.loc[infoStation['INSEE'] == int((currentStation))].index[0]
    infoStation['MeanAnnualRain98-08inmm/yr'][localisation]=AnnualMean
    infoStation['STDRain98-08inmm/yr'][localisation]=AnnualSTD





#%%  Same calulcation, but rather than using complete month, we use complete years only. Done for commparison



plt.close('all')

os.chdir("C:/Users/Reika/Dropbox/PhD_Doc/Precipitation/")


infoStation=pd.read_csv('InfoStattion_12_4.csv', sep=',',encoding="ISO-8859-1")

infoStation['MeanAnnualRain98-08inmm/yr']=np.nan
infoStation['STDRain98-08inmm/yr']=np.nan
 

plt.close('all')

os.chdir("C:/Users/Reika/Dropbox/PhD_Doc/Precipitation/pluvio1998_2008")
#os.chdir("/home/adrien/Desktop/Dropbox/PhD_Doc/Precipitation/pluvio1998_2008")

#files=os.listdir("/home/adrien/Desktop/Dropbox/PhD_Doc/Precipitation/pluvio1998_2008")
files=os.listdir("C:/Users/Reika/Dropbox/PhD_Doc/Precipitation/pluvio1998_2008")


listStation=[]
compil=pd.Series([])

Allcompil=pd.Series([])

monthlyMean=pd.DataFrame

#parcours les fichiers contenus dans le dossier DonneDebitMoyenJournalier
for i in files :
        
    Mois=[]
    MonthlyInterAnnualMean=[]
    MonthlyInterAnnualSTD=[]
    MonthlyInterAnnualCV=[]
    
    DataCurrentStation=pd.read_csv(i, sep=',',encoding="ISO-8859-1")

    
    currentStation=DataCurrentStation.POSTE[1]
    listStation.append(currentStation);    

    Tmeasure=pd.to_datetime(DataCurrentStation.DATE).dt.date.tolist();
    Measure=DataCurrentStation.RR;     
    

    CroppedData=pd.DataFrame(index=Tmeasure)                        
        
    CroppedData['Measure'] = Measure.values;
    CroppedData.index = pd.to_datetime(CroppedData.index)
    
    
    CroppedData['month']=CroppedData.index.month
    CroppedData['year']=CroppedData.index.year
    
    # mean and std 
    rain_an=CroppedData.groupby(CroppedData.index.year).Measure.sum()
        



    localisation=infoStation.loc[infoStation['INSEE'] == int((currentStation))].index[0]
    infoStation['MeanAnnualRain98-08inmm/yr'][localisation]=np.mean(rain_an)
    infoStation['STDRain98-08inmm/yr'][localisation]=np.std(rain_an)


