# scef-model

The script "scef_model.py" implements the Statistical Climate Ensemble Forecast (SCEF) model. 
The implementation herein produces retrospective forecasts of cool-season (November-March) 
CONUS precipitation for the validation years 2000/2001-2019/2020. This file is designed to be 
run interactively as a script. For example, after starting Python, run from your prompt: 

">>>exec(open("scef_model.py",'r').read())"

or from an IPython prompt:

">>>%run -i scef_model.py"

The script will automatically download the predictor data sets if you do not have them already. 
The additional files in the repository provide the necessary basin shapefiles,  historical 
precipitation data, and ensemble mean forecasts of the NMME and ECMWF models.   
