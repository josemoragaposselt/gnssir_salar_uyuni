# gnssir_salar_uyuni

Soil moisture analysis for the Salar de Uyuni in Bolivia using the GNSS-IR library (https://github.com/kristinemlarson/gnssrefl).

This study are the resutls of a study that I carried out for the module: M27-GNSS Environment Sensing of the master in *Geodäsie und Geoinformation* of the University of Bonn. 

## GNSS IR

https://github.com/kristinemlarson/gnssrefl

## GNSS-IR for Soil Moiture Analysis

The Salar is located in Bolivia, on the Andes Mountains at 3656 m over the sea level. The summer weather is charaterized by a rainy season and temperatrues over 0 degrees. During this season the salar surface can be cover by a layer of water. In winter the temperatures goes below zero and there is no precipitation. Snow accumulation is not common because of the ground type. 

The Incacahuasi island is located in the south of the salar at 80 km of the uyuni airport and a GNSS station was is operation till 2017 which observations were handle by the UNAVCO Data Center (https://www.unavco.org/data/gps-gnss/gps-gnss.html). This measurements are supported by the GNSS-IR library  and a pre-process of the data is no necessary.

In order to validate the results provided by the GNSS-IR analysis, I got soil moisture measurements from the SMAP satellite mision (https://smap.jpl.nasa.gov/), which measures using the L3 band with a spatial resolution of 9 km. The processed data for a period of time of 6 months has a size of 89 GB and was processed using a python script provided in this project. A *json* file is available with all the soil mositure measurements (AM and PM) for the period of time analised in this study. Due to the resolution of the mision, as the validataion values are the mean soil moisture value over the whole surface of the salar (10000 km2). 

Precipitation data for the year 2017 was collected from the *national service of meteorology and hydrology of Bolivia* (https://senamhi.gob.bo/index.php/sysparametros). The weather station *Uyuni Aeropuerto* is located 80 km away from the GNSS station. Due to the large distance, this station only provides a reference about the weather conditions during the period of observation.

## Analysis

A presentation of the analysis is shown in the file: GNSS_IR_uyuni_explanation.pdf

Here is possible to see a correlation between the reflected signals and the soil moistures values. However the results does not amtch perfectly because of different reasons:

- accumulation of water on the salar surface -> SMAP measurements goes to their maximums values
- the resolution of the SMAP mision is not representative for the area of study. Soil moisture measurement must be carry out on the posisiton of the GNSS antenna to get representative values and calculate the relationship between the amplitude of the signals and the soil mositure values.
- The weather station is located 80 km away from the GNSS station and it is not representative for all the weather events on the GNSS station

## More GNSS-IR Applications

The code implemented for carring out this study was taken from: https://www.unavco.org/gitlab/gnss_reflectometry/gnssrefl_jupyter

## acknowledgment

- Kristine M. Larson

