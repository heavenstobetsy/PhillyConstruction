# Philly Permits, Construction, and Inspections

&nbsp;&nbsp;&nbsp;&nbsp;See my [Medium article for a more detailed write-up.](https://medium.com/@_heavenstobetsy/predicting-unsafe-housing-in-philadelphia-with-machine-learning-models-d1a364270a9c)


&nbsp;&nbsp;&nbsp;&nbsp;This project uses Philadelphia permit, construction, inspection, and violation data to create a classification model in order to predict unsafe buildings.  

&nbsp;&nbsp;&nbsp;&nbsp;My aim was to create a robust model which actively uses the Philadelphia Licenses and Inspections (L&I) permit application process as a monitoring system for potentially dangerous buildings. [The summary page contains a high level overview of the analysis.](https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Summary.md) For a more technical overview, I've broken up my analysis into separate notebooks:

&nbsp;

</p>

&nbsp;&nbsp;&nbsp;&nbsp; * **_Part 1_** contains the imports needed to run the project.

&nbsp;&nbsp;&nbsp;&nbsp; * **_Part 2_** uses Open Data Philly's API to download data from four separate data sources: construction permits,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; building inspections, housing code violations, and unsafe housing code violations. 

&nbsp;&nbsp;&nbsp;&nbsp; * **_Part 3_** contains the extensive data cleaning and wrangling needed to run the analysis and models. The &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   [My data pre-processing article](https://medium.com/@_heavenstobetsy/predicting-unsafe-housing-in-philadelphia-data-pre-processing-42e13bf72c8d) contains an overview of the data cleaning and blending process.

&nbsp;&nbsp;&nbsp;&nbsp; * **_Part 4_** contains the data modeling process and validation.  The WIP [data model page](https://github.com/heavenstobetsy/PhillyConstruction/blob/master/data%20model.md) will contain an overview of &nbsp;&nbsp;&nbsp;&nbsp;the data modeling process.
