# -*- coding: utf-8 -*-

__help__="""
    This script implements the Statistical Climate Ensemble Forecast (SCEF) 
    model. The implementation herein produces retrospective forecasts of 
    cool-season (November-March) CONUS precipitation for the validation years 
    2000/2001-2019/2020. This file is designed to be run interactively 
    as a script. For example, after starting Python, run from your prompt: 
    >>>exec(open("scef_model.py",'r').read())
    or from an IPython prompt:
    >>>%run -i scef_model.py
"""
__author__ = "Matt Switanek" 
__copyright__ = "Copyright 2021, Matt Switanek" 

##### Read in the necessary Python packages
import json
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import scipy.stats as stats
plt.ion()


###############################################################################
##### Prior to running the model the first time, you can obtain the  
#####   ERSST sea surface temperature data and the NCEP/NCAR reanalysis 
#####   sea level pressure, zonal wind, and the meridional wind data 
#####   sets by uncommenting the following lines of code.
##############################################################################

### Change the following line to "downloadPredictorData = False" if you already 
###     have obtained the predictor data.
downloadPredictorData = True
if downloadPredictorData == True:
    if os.path.exists("predictor_data") == False:
        os.mkdir("predictor_data")
    import ftplib
    ftp = ftplib.FTP("ftp.cdc.noaa.gov")
    ftp.login()
    ftp.cwd("~/Datasets/noaa.ersst.v5")
    ftp.retrbinary("RETR sst.mnmean.nc", open("predictor_data/sst_mon_mean.nc", 'wb').write)
    ftp.cwd("~/Datasets/ncep.reanalysis.derived/surface")
    ftp.retrbinary("RETR slp.mon.mean.nc", open("predictor_data/slp_mon_mean.nc", 'wb').write)
    ftp.cwd("~/Datasets/ncep.reanalysis.derived/pressure")
    ftp.retrbinary("RETR uwnd.mon.mean.nc", open("predictor_data/uwnd_mon_mean.nc", 'wb').write)
    ftp.retrbinary("RETR vwnd.mon.mean.nc", open("predictor_data/vwnd_mon_mean.nc", 'wb').write)
    ftp.quit()


################################################################################
###### Read in observed monthly precipitation data aggregated 
###### to hydrologic units (division 4)
################################################################################

if os.path.exists("forecasts") == False:
    os.mkdir("forecasts")

### This data set provides HUC4 shapefile coordinates along with associated 
###     basin names and numbers.
f1 = open("huc4NumbersNamesShapesLonlat.json",'r')
hucNNSC = json.load(f1)
f1.close()
huc4numbers = hucNNSC['huc4numbers']
huc4names = hucNNSC['huc4names']
huc4shapes = []
for huc in range(0,len(hucNNSC['huc4numbers'])):
    huc4shapes.append(np.float32(hucNNSC['huc4shapes'][huc]))
huc4centroids = np.float32(hucNNSC['huc4centroids'])

### This is the monthly precipitation data aggregated to the HUC4 resolution.
f1 = np.load("huc4precipPRISM_190101-202003.npz")
precipHUC4 = f1['precip']
precip_timeseries = f1['yrMonTimeSeries']
f1.close()

### This is the November-March accumulated precipitation at the HUC4 resolution 
###     for 1901/1902-2019/2020.
coolSeasonPrecip = np.zeros((119,precipHUC4.shape[1]))
for yr in range(0,coolSeasonPrecip.shape[0]):
    coolSeasonPrecip[yr,:] = np.sum(precipHUC4[10+yr*12:15+yr*12,:],axis=0)


###############################################################################
###############################################################################
##### This next section runs the combined lead sea surface temperature (CLSST) model.
###############################################################################
###############################################################################

###############################################################################
###### Set model parameters
###############################################################################

### The number of HUC4 basins across CONUS.
hucNum = 204
### Number of leading principal components of our precipitation HUC4 field to predict.
numPcsPredictand = 5
### Number of preceding lead-time months used to forecast precipitation.
maxlag = 18
### For the CLSST model, the first 99 years are used for calibration, and the 
###     last 20 years are used for validation.
calib1 = np.arange(0,99)
valid1 = np.arange(99,119)

### Break the precipitation data into the calibration and validation periods.
coolSeasonPrecipCalib1 = np.array(coolSeasonPrecip[calib1,:])
coolSeasonPrecipValid = np.array(coolSeasonPrecip[valid1,:])

###############################################################################
###### Read in observed sea surface temperature data data which starts January 1854.
###############################################################################

f1 = netCDF4.Dataset("predictor_data/sst_mon_mean.nc")
### The sst data begin on January, 1854. Therefore, we offset the time to 
###     obtain data begining in January 1899. 
sst = np.array(f1.variables['sst'][540:,:,:]) ###1854-1898 = 45 completed years, then 45x12=540
lat = np.array(f1.variables['lat'])
lon = np.array(f1.variables['lon'])
f1.close()
### Set NaN flagged values to NaN
sst[sst<-5] = np.nan

### Take the average SSTs over the NINO3.4 domain
sstPredictor = np.nanmean(sst[:,42:47,95:120],axis=(1,2)).reshape((sst.shape[0],1)) 

### Build an time series array that has the year and months of our sst array.
sst_timeseries = np.zeros((sstPredictor.shape[0],2),dtype=np.int16)
for yr in range(0,int(sstPredictor.shape[0]/12)):
    for mon in range(0,12):
        sst_timeseries[yr*12+mon,0] = yr+1899
        sst_timeseries[yr*12+mon,1] = mon+1
for mon in range(0,sstPredictor.shape[0]%12):
    sst_timeseries[yr*12+12+mon,0] = yr+1899+1
    sst_timeseries[yr*12+12+mon,1] = mon+1

###############################################################################
###### Find starting months of predictands and the most recent available predictor month
###############################################################################

### Here, we obtain the index locations of the predictor months with a 
###     lead-time of one month. E.g. use September SSTs to forecast Nov-Mar precip.
fnd1 = np.nonzero(sst_timeseries[:,0] == 1901)[0]
fnd2 = np.nonzero(sst_timeseries[fnd1,1] == 9)[0][0]
fndPredictorMonths = np.arange(fnd1[fnd2],fnd1[fnd2]+119*12,12)

### Calculate the mean precipitation at each HUC during the calibation period.
precipMean = np.mean(coolSeasonPrecipCalib1,axis=0)
### Subtract the mean precipitation from the precipitation in the calibration 
###     and the validation periods.
precipCalibZ = np.array(coolSeasonPrecipCalib1-precipMean)
precipValidZ = np.array(coolSeasonPrecipValid-precipMean)


###############################################################################
###### Obtain forecasts using all preceding number of months of SSTs up to the 
######  "maxlag" parameter set above.
###############################################################################

### Build an array to detrend the SST predictor time series 
xx = np.ones((119,2))
xx[:,0] = np.arange(0,119)

### Predicted values for all "maxlag" preceding months
predictedPrecipAllLeads = np.zeros((119,hucNum,maxlag)) 
### Anomaly correlation values that are calculated from the calibration period 
###     for all "maxlag" preceding months
corrCoefs = np.zeros((hucNum,maxlag))

### Iterate over 1-18 months lead-time for our SST NINO3.4 predictor, and fit 
###     a linear regression model separately for each month using the data in 
###     the calibration period.   
for leadtime in range(0,maxlag):
    ### Obtain SST time series for the given lead time. For example use all of 
    ###     the preceding September NINO3.4 values as predictors. 
    sstTS = np.array(sstPredictor[fndPredictorMonths-leadtime,:])
    sstMean = np.mean(sstTS[calib1,:],axis=0)
    sstTSZ = sstTS-sstMean
    
    ### Get predictor values in the validation period as a detrended value with 
    ###     respect to the available preceding time series.
    regCoefs = np.linalg.lstsq(xx[calib1,:],sstTSZ[calib1,0],rcond=None)[0]
    sstTSZ[calib1,0] = sstTSZ[calib1,0]-np.dot(xx[calib1,:],regCoefs)
    for iter1 in range(calib1.shape[0],sstTSZ.shape[0]):
        regCoefs = np.linalg.lstsq(xx[0:iter1+1,:],sstTSZ[0:iter1+1,0],rcond=None)[0]
        sstTSZ[iter1,0] = sstTSZ[iter1,0]-np.dot(xx[iter1,:],regCoefs)

    ### Calculate the left-singular vector, singular values, and the 
    ###     right-singular vector of precipitation data matrix. Do this using 
    ###     the calibration period.
    u1,s1,v1 = np.linalg.svd(precipCalibZ)
    ### Then calculate the principal components of our climate variable field. 
    pcsPrecipCalib = np.dot(precipCalibZ,v1.T)
    
    ### Build predictor and pad with ones to fit with linear regression.
    sstTSZX = np.append(sstTSZ,np.ones((sstTSZ.shape[0],1)),axis=1)
    ### Fit linear regression model (during calibration period) and obtain coefficients 
    ### for the leading principal components for given lead-time.
    regCoefs = np.linalg.lstsq(sstTSZX[calib1,:]\
        ,pcsPrecipCalib[:,:numPcsPredictand],rcond=None)[0]    
    ### Use model fit to forecast PCs for the entire time period.
    predictedValsPCs = np.dot(sstTSZX,regCoefs)
    ### Retransform the predicted values back to real precipitation data space.
    predictedPrecipAllLeads[:,:,leadtime] \
        = np.dot(predictedValsPCs[:,:numPcsPredictand],v1[:numPcsPredictand,:])
    ### Find the anomaly correlations of the forecasts, for this lead-time, during the calibration period.
    for huc in range(0,hucNum):
        corrCoefs[huc,leadtime] \
            = np.corrcoef(predictedPrecipAllLeads[calib1,huc,leadtime]\
            ,precipCalibZ[calib1,huc])[0,1]

### Set anomaly correlations less than 0.01 equal to 0.01. This allows one to 
###     make a forecast even if all lead-times have a negative skill in the 
###     calibration period.
corrCoefs[corrCoefs<.01] = 0.01

### Get the combined forecasts as a function of anomaly correlations in the 
###     calibration period, for given lead-times.
predictedPrecipSST = np.zeros((predictedPrecipAllLeads.shape[0],hucNum))
corrValidation = np.zeros((hucNum))
for huc in range(0,hucNum):
    predictedPrecipSST[:,huc] = np.nansum(predictedPrecipAllLeads[:,huc,:maxlag] \
        *np.abs(corrCoefs[huc,:maxlag]),axis=1) /np.nansum(np.abs(corrCoefs[huc,:maxlag]))
    corrValidation[huc] = np.corrcoef(predictedPrecipSST[valid1,huc],precipValidZ[:,huc])[0,1]

print("Average HUC4 Anomaly Correlation: "+"%.3f" % (np.mean(corrValidation)))

### Save a numpy file that contains the forecasted and observed precipitation.
np.savez("forecasts/clsstForecasts_simulation1.npz",predictedPrecip=predictedPrecipSST,
    observedPrecip=coolSeasonPrecip,predictedYears=np.arange(1901,2020))


###############################################################################
###############################################################################
##### End of CLSST model.
###############################################################################
###############################################################################



###############################################################################
###############################################################################
##### Begin the SCEF model.
###############################################################################
###############################################################################

###############################################################################
##### Read in predictor data and upscale to 5.0 degree latitude by 7.5 degree 
##### longitude resolution.
###############################################################################

### Read in Sea-level pressure data
f1 = netCDF4.Dataset("predictor_data/slp_mon_mean.nc")
slp = np.array(f1.variables['slp'])
f1.close()
slp = (slp[:,0:-1,:]+slp[:,1:,:])/2
slpUpscaled = np.zeros((slp.shape[0],36,48),dtype=np.float32)
for xx in range(0,48):
    for yy in range(0,36):
        slpUpscaled[:,yy,xx] = np.mean(slp[:,yy*2:yy*2+2,xx*3:xx*3+3],axis=(1,2))

### Read in Zonal wind data at 850 hPa
f1 = netCDF4.Dataset("predictor_data/uwnd_mon_mean.nc")
uwnd = np.array(f1.variables['uwnd'][:,2,:,:]) 
f1.close()
uwnd = (uwnd[:,0:-1,:]+uwnd[:,1:,:])/2
uwndUpscaled = np.zeros((uwnd.shape[0],36,48),dtype=np.float32)
for xx in range(0,48):
    for yy in range(0,36):
        uwndUpscaled[:,yy,xx] = np.mean(uwnd[:,yy*2:yy*2+2,xx*3:xx*3+3],axis=(1,2))

### Read in Meridional wind data at 850 hPa
f1 = netCDF4.Dataset("predictor_data/vwnd_mon_mean.nc")
vwnd = np.array(f1.variables['vwnd'][:,2,:,:]) 
f1.close()
vwnd = (vwnd[:,0:-1,:]+vwnd[:,1:,:])/2
vwndUpscaled = np.zeros((vwnd.shape[0],36,48),dtype=np.float32)
for xx in range(0,48):
    for yy in range(0,36):
        vwndUpscaled[:,yy,xx] = np.mean(vwnd[:,yy*2:yy*2+2,xx*3:xx*3+3],axis=(1,2))

### Calculate August-September averages of SLP for the period 1948-2019
predictorSLP = np.zeros((72,slpUpscaled.shape[1],slpUpscaled.shape[2]))
for yr in range(0,predictorSLP.shape[0]):
    predictorSLP[yr,:,:] = np.mean(slpUpscaled[7+yr*12:9+yr*12,:,:],axis=0)

### Calculate August-September averages of UWND for the period 1948-2019
predictorUWND = np.zeros((72,uwndUpscaled.shape[1],uwndUpscaled.shape[2]))
for yr in range(0,predictorUWND.shape[0]):
    predictorUWND[yr,:,:] = np.mean(uwndUpscaled[7+yr*12:9+yr*12,:,:],axis=0)

### Calculate August-September averages of VWND for the period 1948-2019
predictorVWND = np.zeros((72,vwndUpscaled.shape[1],vwndUpscaled.shape[2]))
for yr in range(0,predictorVWND.shape[0]):
    predictorVWND[yr,:,:] = np.mean(vwndUpscaled[7+yr*12:9+yr*12,:,:],axis=0)

### Calculate weights for the predictor data as a function of latitude.
lon20,lat15 = np.meshgrid(np.arange(3.75,360,7.5),np.arange(87.5,-90,-5))
wgts15 = np.cos(np.deg2rad(lat15))          

###############################################################################
##### Get a subset of the precipitation cool season data corresponding to 
#####   the overlap with NCEP/NCAR Reanalysis data set. That is 1948/1949 onward.
###############################################################################
coolSeasonPrecipSub = np.array(coolSeasonPrecip[47:,:])

### For the SLP, UWND, and VWND models, use the first 52 years for calibration, 
###     and the last 20 years are used for validation.
calib2 = np.arange(0,52)
valid2 = np.arange(52,72)

### Cool season precipitation data for the calibration period. 
coolSeasonPrecipCalib2 = np.array(coolSeasonPrecipSub[calib2,:])

###############################################################################
##### Make forecasts using SLP,UWND,VWND fields as predictors
###############################################################################

### Create arrays to fill with forecasts. The dimensions of the following 
###     arrays are (years,hucs,predictorPCs,nlat,slat), where the predictorPCs, 
###     nlat, and slat are our adjustable model parameters. 
###     The "pp" is simply short-hand for predicted precipitation.
ppSLP = np.zeros((72,hucNum,25,4,5),dtype=np.float32) 
ppSLP[:] = np.nan
ppUWND = np.zeros((72,hucNum,25,4,5),dtype=np.float32)
ppUWND[:] = np.nan
ppVWND = np.zeros((72,hucNum,25,4,5),dtype=np.float32)
ppVWND[:] = np.nan

### Loop over our different predictor fields metVar. 
###     where 0 is SLP, 1 is UWND, and 2 is VWND.
for metVar in range(0,3):
    ### Loop over our parameter for the number of leading PCs to use.
    for numPcsPredictor in range(1,26):
        ### Loop over our parameter specifying the northernmost latitude.
        for northLat in range(0,4):
            ### Loop over our parameter specifying the southernmost latitude.
            for southLat in range(15,20):
            
                ### If metVar == 0, use SLP
                ### If metVar == 1, use UWND
                ### If metVar == 2, use VWND
                ### Reshape the array to be 2 dimensional (time, grid points)
                if metVar == 0:
                    predictorData = np.array(predictorSLP[:,northLat:southLat,:]\
                        .reshape(predictorSLP.shape[0],(southLat-northLat)\
                        *predictorSLP.shape[2]))
                if metVar == 1:
                    predictorData = np.array(predictorUWND[:,northLat:southLat,:]\
                        .reshape(predictorUWND.shape[0],(southLat-northLat)\
                        *predictorUWND.shape[2]))
                if metVar == 2:
                    predictorData = np.array(predictorVWND[:,northLat:southLat,:]\
                        .reshape(predictorVWND.shape[0],(southLat-northLat)\
                        *predictorVWND.shape[2]))

                ### Get the latitudinal-derived weights to apply to our predictor matrix
                currWgts = np.array(wgts15[northLat:southLat,:]).flatten()

                ### Iterate over the years in the length of the NCEP/NCAR data set.
                ###     For years in the calibration period, we perform leave-one-out 
                ###     cross-validation. Then, for the last period (last 20 years), 
                ###     all of the data in the calibration period is used to fit 
                ###     the model. 
                for yr in range(0,72):
                    fndYr = np.nonzero(calib2==yr)[0]
                    calibSub = np.delete(calib2,fndYr)
                    
                    ### Detrend the individual predictor time series using the 
                    ###     calibration period excluding the current year.
                    xx = np.ones((predictorData.shape[0],2))
                    xx[:,0] = np.arange(0,predictorData.shape[0])
                    regCoefs = np.linalg.lstsq(xx[calibSub,:],predictorData[calibSub,:],rcond=None)[0]
                    predictorDataDetrended = np.array(predictorData-np.dot(xx,regCoefs))

                    ### Weight the predictor data as a function of latitude.
                    predictorDataDetrendedZ = (predictorDataDetrended-np.mean(predictorDataDetrended[calibSub,:],axis=0)) * currWgts 
                    
                    ### Remove mean from precipitation data in the calibration period.
                    precipDataCalibZ = np.array(coolSeasonPrecipCalib2-np.mean(coolSeasonPrecipCalib2,axis=0))
                    ### Decompose the predictor data and the precipitation data 
                    ###     using SVD decomposition.
                    u1,s1,v1 = np.linalg.svd(predictorDataDetrendedZ[calibSub,:])
                    u2,s2,v2 = np.linalg.svd(precipDataCalibZ[calibSub,:])
                    ### Use the right-singular vectors to obtain our principal 
                    ###     components of the predictors and predictands.
                    pcsX = np.dot(predictorDataDetrendedZ,v1.T)
                    pcsCalibY = np.dot(precipDataCalibZ,v2.T)
                    
                    ### Create an array with the number of leading PCs of our 
                    ###     predictor data that we will use, padded with ones. 
                    ###     This will be used for fitting our multiple linear 
                    ###     regression model.
                    xDataPadded = np.zeros((pcsX.shape[0],numPcsPredictor+1))
                    xDataPadded[:,:-1] = pcsX[:,:numPcsPredictor]

                    ### Get the regression coefficients from our principal component 
                    ###     mulitple linear regression model.
                    regCoefs = np.linalg.lstsq(xDataPadded[calibSub,:],pcsCalibY[calibSub,:numPcsPredictand],rcond=None)[0]
                    ### Calculate the leading PCs of our precipitation for the 
                    ###     current left-out year.
                    predictedNowPCs = np.dot(xDataPadded[yr,:],regCoefs)

                    if metVar == 0:
                        ### If metVar == 0, fill ppSLP. Here we also backtransform 
                        ###     the predicted PCs to real precipitation data space.
                        ppSLP[yr,:,numPcsPredictor-1,northLat,southLat-15] = np.dot(predictedNowPCs,v2[:numPcsPredictand,:])
                    if metVar == 1:
                        ### If metVar == 1, fill ppUWND. Here we also backtransform 
                        ###     the predicted PCs to real precipitation data space.
                        ppUWND[yr,:,numPcsPredictor-1,northLat,southLat-15] = np.dot(predictedNowPCs,v2[:numPcsPredictand,:])
                    if metVar == 2:
                        ### If metVar == 0, fill ppVWND. Here we also backtransform 
                        ###     the predicted PCs to real precipitation data space.
                        ppVWND[yr,:,numPcsPredictor-1,northLat,southLat-15] = np.dot(predictedNowPCs,v2[:numPcsPredictand,:])

                
        print(time.asctime(),metVar,numPcsPredictor)

### Save a numpy file that contains the forecasted precipitation for the different 
###     predictors and observed precipitation. 
np.savez("forecasts/slp_uwnd_vwnd_predictLast20years_simulation1.npz",predicted_slp=ppSLP\
    ,predicted_uwnd=ppUWND,predicted_vwnd=ppVWND,observedPrecip=coolSeasonPrecipSub\
    ,predictedYears=np.arange(1948,2020))


### Ultimately use the precpitation data from 1948/1949-2019/2020 as 
###     our observational period of record. This is because this is the period 
###     of the forecasts for the SCEF model. 
observedPrecip = np.array(coolSeasonPrecipSub)

### Reshape our precipitation predictions from the SCEF model to be 
###     (time,space,all possible parameter combinations). In our case, here, 
###     there are 500 parameter combinations.
ppSLPR = np.array(np.reshape(ppSLP,(72,204,-1)))
ppUWNDR = np.array(np.reshape(ppUWND,(72,204,-1)))
ppVWNDR = np.array(np.reshape(ppVWND,(72,204,-1)))

### Get z-scores with respect to the calibration period. 
ppSLPRZ = (ppSLPR-np.mean(ppSLPR[calib2,:,:],axis=0))/np.std(ppSLPR[calib2,:,:],axis=0)
ppUWNDRZ = (ppUWNDR-np.mean(ppUWNDR[calib2,:,:],axis=0))/np.std(ppUWNDR[calib2,:,:],axis=0)
ppVWNDRZ = (ppVWNDR-np.mean(ppVWNDR[calib2,:,:],axis=0))/np.std(ppVWNDR[calib2,:,:],axis=0)
observedPrecipZ = (observedPrecip-np.mean(observedPrecip[calib2,:],axis=0))/np.std(observedPrecip[calib2,:],axis=0)

### Calculate the correlation coefficients over the calibration period
###     for each HUC and each parameter combination. 
ccSLP = np.zeros((204,ppSLPR.shape[2]))
ccUWND = np.zeros((204,ppSLPR.shape[2]))
ccVWND = np.zeros((204,ppSLPR.shape[2]))
for huc in range(0,204):
    for ensNum in range(0,500):
        ccSLP[huc,ensNum] = np.corrcoef(ppSLPR[calib2,huc,ensNum],observedPrecip[calib2,huc])[0,1]
        ccUWND[huc,ensNum] = np.corrcoef(ppUWNDR[calib2,huc,ensNum],observedPrecip[calib2,huc])[0,1]
        ccVWND[huc,ensNum] = np.corrcoef(ppVWNDR[calib2,huc,ensNum],observedPrecip[calib2,huc])[0,1]

### Save a numpy file that contains the correlation coefficients for 
###     each HUC and for each parameter combination.
np.savez("forecasts/slp_uwnd_vwnd_predictLast20years_hcorr_simulation1.npz",\
    ccSLP=ccSLP,ccUWND=ccUWND,ccVWND=ccVWND)
    
### This is the number of best performing ensemble members that we will use 
###     to calculate an ensemble mean forecast. This number is 1% of the 500 
###     parameter combinations used here in this script.
numEns2use = 5

### Get the CONUS-average correlation coefficients (average across all 204 HUCs),
###     then obtain the sorted indices of the worst-to-best performing parameter 
###     combinations, as they performed in the calibration period.
srtMod1 = np.argsort(np.mean(ccSLP[:,:],axis=0))
srtMod2 = np.argsort(np.mean(ccUWND[:,:],axis=0))
srtMod3 = np.argsort(np.mean(ccVWND[:,:],axis=0))
### Calculate the ensemble mean predictions for each submodel (SLP, UWND, VWND) 
###     over the calibration period.
ppSLPem1 = np.mean(ppSLPR[calib2,:,:][:,:,srtMod1[-numEns2use:]],axis=(2))
ppUWNDem1 = np.mean(ppUWNDR[calib2,:,:][:,:,srtMod2[-numEns2use:]],axis=(2))
ppVWNDem1 = np.mean(ppVWNDR[calib2,:,:][:,:,srtMod3[-numEns2use:]],axis=(2))
### Calculate the ensemble mean predictions for each submodel (SLP, UWND, VWND) 
###     over the validation period.
ppSLPem2 = np.mean(ppSLPR[valid2,:,:][:,:,srtMod1[-numEns2use:]],axis=(2))
ppUWNDem2 = np.mean(ppUWNDR[valid2,:,:][:,:,srtMod2[-numEns2use:]],axis=(2))
ppVWNDem2 = np.mean(ppVWNDR[valid2,:,:][:,:,srtMod3[-numEns2use:]],axis=(2))
### Combine the forecasts from the calibration and validation periods.
ppSLPem = np.append(ppSLPem1,ppSLPem2,axis=0)
ppUWNDem = np.append(ppUWNDem1,ppUWNDem2,axis=0)
ppVWNDem = np.append(ppVWNDem1,ppVWNDem2,axis=0)

### Get our forecasts from the CLSST model over the same period of time as the SCEF model.
ppSST = np.array(predictedPrecipSST[47:,:])

### Merge the forecasts across the four individual submodels as a function 
###     of their correlation in the calibration period. 
ppMERGED = np.zeros((72,204))
#### Obtain Z-scores of the forecasts at each HUC with respect to the calibration period.
ppSSTZ = (ppSST-np.mean(ppSST[calib2,:],axis=0))/np.std(ppSST[calib2,:],axis=0)
ppSLPZ = (ppSLPem-np.mean(ppSLPem[calib2,:],axis=0))/np.std(ppSLPem[calib2,:],axis=0)
ppUWNDZ = (ppUWNDem-np.mean(ppUWNDem[calib2,:],axis=0))/np.std(ppUWNDem[calib2,:],axis=0)
ppVWNDZ = (ppVWNDem-np.mean(ppVWNDem[calib2,:],axis=0))/np.std(ppVWNDem[calib2,:],axis=0)
for huc in range(0,204):
    r1 = ((np.corrcoef(ppSSTZ[calib2,huc],observedPrecip[calib2,huc])[0,1]+1.)/2)**2
    r2 = ((np.corrcoef(ppSLPZ[calib2,huc],observedPrecip[calib2,huc])[0,1]+1.)/2)**2
    r3 = ((np.corrcoef(ppUWNDZ[calib2,huc],observedPrecip[calib2,huc])[0,1]+1.)/2)**2
    r4 = ((np.corrcoef(ppVWNDZ[calib2,huc],observedPrecip[calib2,huc])[0,1]+1.)/2)**2        
    ppMERGED[:,huc] = (r1*ppSSTZ[:,huc]+r2*ppSLPZ[:,huc]+r3*ppUWNDZ[:,huc]+r4*ppVWNDZ[:,huc])/(r1+r2+r3+r4)

### Assume that the forecasts in the out-of-sample period are unbiased.
ppMERGED[valid2,:] = (ppMERGED[valid2,:]-np.mean(ppMERGED[valid2,:],axis=0))/np.std(ppMERGED[valid2,:],axis=0)

np.savez("forecasts/mergedForecasts1.npz"\
    ,predictedMERGED=ppMERGED,observedPrecip=observedPrecip\
    ,predictedSST=ppSSTZ,predictedSLP=ppSLPZ\
    ,predictedUWND=ppUWNDZ,predictedVWND=ppVWNDZ)

###############################################################################
###############################################################################
##### End of SCEF model.
###############################################################################
###############################################################################


###############################################################################
###############################################################################
##### Calculate CONUS-average anomaly correlations over the validation period 
#####       2000/2001-2019/2020 for SCEF, NMME, and ECMWF.
###############################################################################
###############################################################################

#####f1 = np.load("forecasts/mergedForecasts1.npz")
#####ppMERGED = f1['predictedMERGED']
#####f1.close()

### Read in the NMME forecasts for the HUC4 basins
f1 = np.load("NMME_pr_hucs_NDJFM_1982t1983-2020t2021.npz")
precipNMMEhucs = f1['precipNMMEhuc'][:-1,:]
f1.close()

### Read in the ECMWF forecasts for the HUC4 basins
f1 = np.load("ECMWF_pr_hucs_NDJFM_1994t1995-2020t2021.npz")
precipECMWFhucs = f1['precipECMWFhuc'][:-1,:]
f1.close()

### Calculate the anomaly correlations for the validation period for all HUCs 
###     using the SCEF model
acSCEF = np.zeros((204))
for huc in range(0,204):
    acSCEF[huc] = np.corrcoef(ppMERGED[52:,huc],observedPrecip[52:,huc])[0,1]

### Calculate the anomaly correlations for the validation period for all HUCs 
###     using the NMME model
acNMME = np.zeros((204))
for huc in range(0,204):
    acNMME[huc] = np.corrcoef(precipNMMEhucs[-20:,huc],observedPrecip[52:,huc])[0,1]

### Calculate the anomaly correlations for the validation period for all HUCs 
###     using the ECMWF model
acECMWF = np.zeros((204))
for huc in range(0,204):
    acECMWF[huc] = np.corrcoef(precipECMWFhucs[-20:,huc],observedPrecip[52:,huc])[0,1]

### Print the 
print("SCEF CONUS-AVG = %.3f" % (np.mean(acSCEF)))
print("NMME CONUS-AVG = %.3f" % (np.mean(acNMME)))
print("ECMWF CONUS-AVG = %.3f" % (np.mean(acECMWF)))



