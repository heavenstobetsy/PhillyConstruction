
## Building the Data
&nbsp;&nbsp;&nbsp;&nbsp;This model is meant to act as a tool for Philadelphia's Department of Licenses and Inspections to prioritize which buildings to investigate after a building permit is submitted. Specifically, I wanted to figure out which buildings would be most likely be declared unsafe and extremely hazardous - where immediate action needs to be taken. For a less technical summary, along with my motivation and reasoning for choosing this problem, see
[my Medium write-up](https://medium.com/@_heavenstobetsy/predicting-unsafe-housing-in-philadelphia-with-machine-learning-models-d1a364270a9c).
<p>
 &nbsp;
    </p>
    
## Raw Datasets
&nbsp;&nbsp;&nbsp;&nbsp;For this model I used data from The City of Philadelphia's Department of Licenses and Inspections. I focused on buildings with a permit submitted up to June 2019 in order to keep the data manageable; I also wanted to collect an extra 6+ months of code violation/unsafe status data after the last permit date - having the same date for the permit cutoff and the unsafe status cutoff might result in erroneous data.

&nbsp;&nbsp;&nbsp;&nbsp;The L&I data was built into a dataframe comprised of several datasets, which includes data about permits, inspections, code violations, and unsafe violations. The four separate data sources are summarized below:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Permits:* data on building construction/use/update permits. <p></p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Inspections:* data on L&I inspection cases and compliance/non-compliance follow-up.. <p></p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Violations:* data and descriptions of every building code violation, and risk level. <p></p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Unsafe Violations:* data and descriptions of the unsafe building code violations. <p></p>
&nbsp;
&nbsp;
<p>
<p>
The chart below shows the breakdown of the total number of buildings for each type of data. There are ~80,000 buildings with submitted permits, and (thankfully) relatively few unsafe buildings.
  
![Building Counts](https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/unique_counts.png)
[Fig. 1 – Bar Chart of 2019 Building Counts]
<p>
 &nbsp;
    </p>
    
## Data Wrangling

&nbsp;&nbsp;&nbsp;&nbsp;When aggregating and merging these massive datasets, I had to be careful to not include future inspections and violations past the permit submission date, along with other data that might bleed into and influence the model.

&nbsp;&nbsp;&nbsp;&nbsp;Beyond that, I needed to summarize the datasets on multiple levels and ultimately join them to the permit/address level - both parts took up the majority of my time on this project, due to the sheer size of the data, along with errors and messiness in the data.

&nbsp;&nbsp;&nbsp;&nbsp;To connect the data, I first joined the permit and inspection datasets on a unique ID (addresskey), which is used for a unique address/owner combination. After that, I cleaned the data and removed erroneous entries, weird data, and combined L&I's outdated segmentation.

&nbsp;&nbsp;&nbsp;&nbsp;After building a clean, aggregated file for permit and inspection data, I then moved on to cleaning, aggregating, and merging code violation data - and then the same with unsafe building data.

&nbsp;&nbsp;&nbsp;&nbsp;I was hoping to add in property complaint and violation fee data, but (probably for good reason) building complaints and fees are generalized as street blocks, and not actual addresses. Still, that didn't prevent me from trying to connect the datasets using GPS coordinates and various infraction/case resolution dates. However, I was ultimately unsuccessful (again, probably for the best).
<p>
 &nbsp;
    </p>
    
## Feature Engineering
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In the real world, improving models using feature engineering can be crucial - feature engineering can often improve accuracy, and turn an average model into something worth keeping. For this problem, I manually created new attributes that I believed would be predictive of a building's unsafe status. In one case, I used a flagging system, along with a data cleanup to create a new variable that indicated when a contractor was handling the construction, or if a homeowner or another entity was responsible.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;I also looked at grouping common violation descriptions by highly ranked keywords, and calculated the time between a permit submission and the building's most recent inspection date.  Below are ranked feature charts, sorted by importance.

<p>
 &nbsp;
    </p>
    
    
<p float="center">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/feature_importsance.png" width="300" />
  <img src="https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/most_feature_importanes.png" width="300" /> 
	<p>
[Figs. 1&2 – Scaled variable importance among predictors]
	

<p>
 &nbsp;
    </p>
    
## Categorical Variables

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Creating dummy variables vastly increased the number of features in my dataset, which is why selecting important features and pruning out unimportant features for predicting is necessary. When originally exploring the dataset, I found one "permit grouping" category that resulted in 180 dummy variables.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Two categories that I found important were 1) type of permit work, and 2) permit description. Permit work had categories like "electrical" or "plumbing", while permit description which goes into more detail on what type of construction will happen.

See [Categorical Variables](https://gist.github.com/heavenstobetsy/38b48eda46dab9a134b730ebdec7d6c6) for more details.  The next post on the modeling process will go over pruning the features and model selection.

<p>
 &nbsp;
    </p>


   
<p>
<p>
<p>
<p>
<p>
<p>
&nbsp;
&nbsp;
&nbsp;
&nbsp;
&nbsp;
  </p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This project was completed by E. Johnson, and last updated in January 2020. The data I used came from the 
City of Philadelphia Departments of Licenses and Inspections located, at [OpenDataPhilly](https://www.opendataphilly.org/). I conducted the analysis using 
Python and used the following packages: AWS SageMaker, pandas,numpy, sklearn, matplotlib, imbalanced, xgboost, mlens, Boruta, seaborn, and waffle.
