# Processing of point data as input to compare the effects of several paramaters on lanslide occurence (climate / geomorphology / landuse)
# Dependent variable : binary variable representing the presence or absence of landslide
# Independent variables: Data are point from a raster (maps) for different layers
 

#load libraries

library(readxl)
require(ISLR)
library(raster)
library(sp)
library(rgdal)
library(ROCR)
library(pROC)
library(stats)
library(data.table)
library(varhandle)



#read table with all points data
attTable_5 = read.csv("C:/Users/Reika/Dropbox/data/Landslide/tableLSscaledNEW.csv", header = TRUE)

# rename the columns and drop useless ones
setnames(attTable_5, old=c("slopeScaled_Band_1","elevScaled_Band_1","DeepInfScaled_Band_1","ShalowInfScaled_Band_1","runoffScaled_Band_1","rainScaled_Band_1","aspScaled_Band_1","curvScaled_Band_1"), new=c("slope", "elevation", "DeepInf", "ShalInf", "runoff", "pluie", "aspect", "curvature"))
cols.dont.want <- c("OBJECTID", "pointWS","X","Y")
attTable_5 <- attTable_5[, ! names(attTable_5) %in% cols.dont.want, drop = F]

attTable_5 <- na.omit(attTable_5)



################# TEST 7/31/2019    all parameters included

attTable_6 = read.csv("C:/Users/Reika/Dropbox/data/Landslide/Sample_point_8_1.csv", header = TRUE,sep=";")

setnames(attTable_6, old=c("SLOPESCALED","ELEVSCALED","SHALOWINFSCALED","RUNOFFSCALED","RAINSCALED","ASPSCALED","CURVSCALED","RASTER_DEEP_INF_SCALED","RASTER_WSRAINFALL_SCALED"), new=c("slope", "elevation", "ShalInf", "runoff", "pluiePt", "aspect", "curvature", "DeepInf","rainWS"))
cols.dont.want <- c("POINTWS","XCoord","YCoord","X","Y","OBJECTID")
attTable_6 <- attTable_6[, ! names(attTable_6) %in% cols.dont.want, drop = F]

attTable_6["aspectNum"] <- as.numeric(as.character(attTable_6$aspect))
attTable_6["elevNum"] <- as.numeric(as.character(attTable_6$elevation))
attTable_6["curvNum"] <- as.numeric(as.character(attTable_6$curvature))
attTable_6["slopeNum"] <- as.numeric(as.character(attTable_6$slope))
attTable_6["LANDUSE6Num"] <- as.numeric(as.character(attTable_6$LANDUSE6))
attTable_6["LANDUSE5Num"] <- as.numeric(as.character(attTable_6$LANDUSE5))
attTable_6["LANDUSE4Num"] <- as.numeric(as.character(attTable_6$LANDUSE4))
attTable_6["LANDUSE3Num"] <- as.numeric(as.character(attTable_6$LANDUSE3))
attTable_6["LANDUSE2Num"] <- as.numeric(as.character(attTable_6$LANDUSE2))
attTable_6["LANDUSE1Num"] <- as.numeric(as.character(attTable_6$LANDUSE1))

cols.dont.want <- c("aspect","elevation","curvature","slope","LANDUSE6","LANDUSE5","LANDUSE5_1","LANDUSE4","LANDUSE3","LANDUSE2","LANDUSE1")
attTable_6 <- attTable_6[, ! names(attTable_6) %in% cols.dont.want, drop = F]


attTable_6[attTable_6 == "NULL"] = NA

attTable_6 <- na.omit(attTable_6)


summary(attTable_6)


#test mutlicolinearity

x<-attTable_6[,-1]
y<-attTable_6[,1]
omcdiag(x,y)
imcdiag(x,y)

library(ppcor)
corr <- pcor(x, method = "pearson")



fit <- glm(INLS ~  slopeNum + aspectNum + curvNum + elevNum + RECLASS_LANDUSE_1, data=attTable_6 ,family = poisson)
summary(fit)

prob <- predict(fit,newdata=subset(attTable_6,select=c("slopeNum","aspectNum","curvNum","rainWS","elevNum","LANDUSE4","LANDUSE5","LANDUSE1","LANDUSE2","LANDUSE3","LANDUSE6")), type="response")
attTable_6$prob=prob
g2 <- roc(INLS ~ prob, data = attTable_6)
plot(g,col='red') 
print(" Area under curve pour LR model avec infiltration :")
auc(g)

coef<-fit$coefficients
write.table(coef, "C:/Users/Reika/Dropbox/data/Landslide/coef.txt", sep=";")



################# TEST 8/7/2019    Model Test Frequency ratio only

attTableFR = read.csv("C:/Users/Reika/Dropbox/data/Landslide/TabPointFRModelAout19.csv", header = TRUE,sep=",")

attTableFR[attTableFR == "-9999"] = NA


library(pROC)


g1 <- roc(inLS ~ FR_baselin, data = attTableFR)
plot(g1,col='red') 


g <- roc(inLS ~ FR_allWS, data = attTableFR)
lines(g,col='blue') 
  
legend("bottomright", legend=c("BFR", "FR"),
       col=c("red", "blue"), lty=1:1)
  
  auc(g)


  
  ################# TEST 8/8/2019    FR pthe logistic regression
  
  attTable_6 = read.csv("C:/Users/Reika/Dropbox/data/Landslide/TabPointFRLayers.csv", header = TRUE,sep=",")
  
  setnames(attTable_6, old=c("Reclass_slope_FR","Reclass_elev_FR","Reclass_ShallInfWS_FR","Reclass_runoffWS_FR","Reclass_rainPt_FR","Reclass_asp_FR","Reclass_curv_FR","Reclass_DeepInfWS_FR","Reclass_rainfallWS_FR","Reclass_landuse_FR"), new=c("slope", "elevation", "ShalInf", "runoff", "pluiePt", "aspect", "curvature", "DeepInf","rainWS","landuse"))
  cols.dont.want <- c("POINTWS","XCoord","YCoord","X","Y","OBJECTID")
  attTable_6 <- attTable_6[, ! names(attTable_6) %in% cols.dont.want, drop = F]
  
  
  attTable_6[attTable_6 == "NULL"] = NA
  
  attTable_6 <- na.omit(attTable_6)
  
#  scaled.attTable_6 <- scale(attTable_6)
  
  scaledAttTable_6 <- as.data.frame(apply(attTable_6[, 3:12], 2, function(x) (x - min(x))/(max(x)-min(x))))
  scaledAttTable_6["pointWS"] <- attTable_6$pointWS
  scaledAttTable_6$inLS <- attTable_6$inLS

  
  scaledAttTable_6["aspectNum"] <- as.numeric(as.character(scaledAttTable_6$aspect))
  scaledAttTable_6["elevNum"] <- as.numeric(as.character(scaledAttTable_6$elevation))
  scaledAttTable_6["curvNum"] <- as.numeric(as.character(scaledAttTable_6$curvature))
  scaledAttTable_6["slopeNum"] <- as.numeric(as.character(scaledAttTable_6$slope))
  
  summary(scaledAttTable_6)
  
  
  fit <- glm(inLS ~  slope + aspect + aspect + curvature + landuse + elevation + DeepInf + ShalInf + runoff + rainWS, data=scaledAttTable_6 ,family = poisson)
  summary(fit)
  
  prob <- predict(fit,newdata=subset(scaledAttTable_6,select=c("slope","aspect","curvature","landuse","elevation","DeepInf","ShalInf","runoff","rainWS")), type="response")
  scaledAttTable_6$prob=prob
  g <- roc(inLS ~ prob, data = scaledAttTable_6)
  plot(g,col='red') 
  print(" Area under curve pour LR model avec infiltration :")
  auc(g)
  
  coef<-fit$coefficients
  write.table(coef, "C:/Users/Reika/Dropbox/data/Landslide/coef.csv", sep=";")  
  
  
  
  
  #plot diagnostic
  par(mfrow=c(2,2))
  plot(fit)
  
  
  #test mutlicolinearity
#  X <-data.frame(attTable_6$rainWS,attTable_6$DeepInf,attTable_6$LANDUSE6,attTable_6$LANDUSE5,attTable_6$LANDUSE4,attTable_6$LANDUSE3,attTable_6$LANDUSE2,attTable_6$LANDUSE1,attTable_6$ShalInf,attTable_6$runoff,attTable_6$pluiePt,attTable_6$aspectNum,attTable_6$elevNum,attTable_6$curvNum,attTable_6$slopeNum)
  
  newData <-data.frame(scaledAttTable_6$rainWS,scaledAttTable_6$DeepInf,scaledAttTable_6$landuse,scaledAttTable_6$ShalInf,scaledAttTable_6$runoff,scaledAttTable_6$pluiePt,scaledAttTable_6$aspectNum,scaledAttTable_6$elevNum,scaledAttTable_6$curvNum,scaledAttTable_6$slopeNum)
  newData[newData == "NULL"] = NA
  
  newData <- na.omit(newData)
  
  newData <- newData[c("scaledAttTable_6.slopeNum","scaledAttTable_6.aspectNum","scaledAttTable_6.curvNum","scaledAttTable_6.landuse","scaledAttTable_6.elevNum","scaledAttTable_6.DeepInf","scaledAttTable_6.ShalInf","scaledAttTable_6.runoff","scaledAttTable_6.rainWS")]
  
  x<-newData[,-1]
  y<-newData[,1]
  
  library(mctest)
  omcdiag(x,y)
  imcdiag(x,y)
  
  library(ppcor)
  corr <- pcor(x, method = "pearson")
  
  
  # compilation of plot aveclegend
  
  plot(g1,col='red') 
  
  lines(g,col='blue') 
  
  lines(g2,col='green')
  
  lines(g3,col='green',type="l", lty=2)
  
  lines(g4,col='black')
  
  
  legend("bottomright", legend=c("BFR (AUC 0.83)", "FR (AUC 0.81)","BLR2 (AUC 0.78)","BLR1 (AUC 0.78)","LR3 (AUC 0.81)"),
         col=c("red", "blue", "green", "green", "black"), lty=c(1,1,1,2,1))
  
  

