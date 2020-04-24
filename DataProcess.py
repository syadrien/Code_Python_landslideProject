# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:04:08 2017

@author: ASY

Goal: Compilation of available underground physico-chemical data on Reunion Island.
      + classification of stations according to the type of surface/basal aquifer
Input : File RecapDonne.csv which contains a lot of info (some redundant, some measurements of a parameter were taken at different times)
Output : A clean csv file including information on each station + averages of different parameters measured.
         + a second csv file compiling the same info but only for basal aquifer
         
See article Join(1997) for the criterion of differentiation of aquifers according to [Na]/[Cl] ratios.

Translated with www.DeepL.com/Translator (free version)

"""

import os
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import operator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

#os.getcwd()
os.chdir("C:/Users/Reika/Dropbox/donne-ades-Reu-souterrain")

plt.close('all')

dataRaw=pd.read_csv('RecapDonne.csv', sep=';', encoding="ISO-8859-1")
test=dataRaw.groupby(['CodeNationalBSS']).sum().reset_index()



databis=dataRaw.groupby(['CodeNationalBSS']).mean()


#%%

AllForage=databis[databis.Codenaturedupointdeau==6]

AllCaptif=AllForage[AllForage.Codemodedegisement==2]


AllForage['concNabis']=(1/(10**-3*22.99))*AllForage.NaenmgparL;
AllForage['concClbis']=(1/(10**-3*35.45))*AllForage.ClenmgparL;

BasalForage=AllForage[((AllForage.concClbis<10**((1.33)*np.log10(AllForage.concNabis)-1.26))&(AllForage.concClbis<1000))]

#%%

code=databis.index

newTab=pd.DataFrame(index=code)

pH=databis.pH;
newTab['pH'] = pH;

Wtable=databis.CoteNGF;
newTab['Wtable'] = Wtable;
      
WTemp=databis.tempC;
newTab['WTemp'] = WTemp;
      
concCl=databis.ClenmgparL;
newTab['concCl'] = concCl;

concNa=databis.NaenmgparL;
newTab['concNa'] = concNa;
      
X=databis.X_WGS84;
newTab['X'] = X;
      
Y=databis.Y_WGS84;
newTab['Y'] = Y;
      
Z=databis.Altitude;
newTab['Z'] = Z;
      
DO=databis.OxygenedissousenmgparL;
newTab['DO'] = DO;
      
MaxDepth=databis.Profondeurinvestigationmaximale;
newTab['MaxDepth'] = MaxDepth;
#PointState=databis.Etatdupointdeau;
#WellType=databis.Naturedupointdeau;
#AquType=databis.Modedegisement;
concCa=databis.CaenmgparL;
newTab['concCa'] = concCa;
      
concMg=databis.MgenmgparL;
newTab['concMg'] = concMg;
      
concNitrate=databis.NitrateenmgparL;
newTab['concNitrate'] = concNitrate;
      
concSilicate=databis.SilicateenmgparL;
newTab['concSilicate'] = concSilicate;
      
concSilice=databis.SiliceenmgparL;
newTab['concSilice'] = concSilice;
      
Typepointeau=databis.Codetypedupointdeau;
newTab['Typepointeau'] = Typepointeau;
      
Etatpoint=databis.Codeetatpoint;
newTab['Etatpoint'] = Etatpoint;
      
Nature=databis.Codenaturedupointdeau;
newTab['Nature'] = Nature;
      
ModeGisement=databis.Codemodedegisement;
newTab['ModeGisement'] = ModeGisement;
    
      
#data['Altitude']

#plt.figure(1)
#
##CM = plt.cm.get_cmap('jet')
##m = cm.Scalar
##
###Mappable(cmap=cm.jet)
##mbis=m.set_array(Z)
#
#ax = plt.subplot(111, projection='3d')
#sc=ax.scatter(X,Y,Z,c=WTemp)
##fig.colorbar(sc)
#plt.show()
#plt.xlabel("X_WGS84")
#plt.ylabel("Y_WGS84")
#ax.set_zlabel("Z(m)")


concNabis=(1/(10**-3*22.99))*concNa;
concClbis=(1/(10**-3*35.45))*concCl;



#plt.legend(loc='upper left',ncol=3)

xline1=np.arange(100,10000);
yline1=((1.39)*xline1+39);

xline2=np.arange(100,10000);
yline2=10**((1.33)*np.log10(xline2)-1.26);       
       
#plt.figure(2)
#
#plt.loglog(concNabis,concClbis,'r.')
#plt.loglog(xline1,yline1,'b-')
#plt.loglog(xline2,yline2,'g-')
#plt.legend(['valeur','Rainwater limit','Spring limit'])
#plt.xlabel("[Na](mg/L)")
#plt.ylabel("[Cl](mg/L)")

supAqu=(concClbis>10**((1.33)*np.log10(concNabis)-1.26))
basAqu=(concClbis<10**((1.33)*np.log10(concNabis)-1.26))
basAquNotConta=((concClbis<10**((1.33)*np.log10(concNabis)-1.26))&(concClbis<1000))

AllForage=databis[databis.Codenaturedupointdeau==6]

databis['basAquNotConta'] = basAquNotConta;

listBasalAqu=databis[databis.basAquNotConta != False];


ForagebasAquNotConta=listBasalAqu[listBasalAqu.Codenaturedupointdeau==6]
PuitbasAquNotConta=listBasalAqu[listBasalAqu.Codenaturedupointdeau==12]
       
ForagebasAquNotConta.to_csv('ForagebasAquNotConta.csv')
PuitbasAquNotConta.to_csv('PuitbasAquNotConta.csv')
AllForage.to_csv('AllForage.csv')

       
#databis['Depth'] = databis.Altitude-databis.CoteNGF;      

#%%        
        
listBasalAqu=databis[databis.basAquNotConta != False];
#listBasalAqu=databis


listBasalAqu=listBasalAqu.ix[~(listBasalAqu['Profondeurinvestigationmaximale'] < 25)];
##
#listBasalAqu.to_csv('listBasalAqu.csv')