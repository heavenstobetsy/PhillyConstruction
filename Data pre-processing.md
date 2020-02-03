
## Building the Data
&nbsp;&nbsp;&nbsp;&nbsp;This model is meant to act as a tool for Philadelphia’s Department of Licenses and Inspections to 
prioritize which buildings to investigate after a building permit is submitted. Specifically, I wanted to figure out which 
building would be most likely be declared unsafe and extremely hazardess and needs immediate action taken.  For a less 
technical summary, along with my motivation and reasoning for choosing this problem see in my Medium write-up
at [link].
<p>
 &nbsp;
    </p>
    
## Datasets
&nbsp;&nbsp;&nbsp;&nbsp;For this model I used data from The City of Philadelphia’s Department of Licenses and Inspections. 
To keep the data manageable, I focused on buildings with a permit submitted up to through June 2019. The L&I data was
built into a dataframe comprised of several datasets including permits, inspections, code violations, and unsafe violations. 
This includes:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Permits: data on building construction/use/update permits. <p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Inspections: data on cases and compliance/non-compliance follow-up. <p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Violations: data and descriptions of every building code violation, and risk level. <p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Unsafe Violations: data and descriptions of the unsafe building code violations. <p>
&nbsp;
&nbsp;
<p>
<p>
This chart shows the breakdown of the total number of buildings for each type of data:
  
![Building Counts](https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/unique_counts.png)
[Fig. 1 – Bar Chart of 2019 Building Counts]
<p>
 &nbsp;
    </p>
    
## Data Wrangling

&nbsp;&nbsp;&nbsp;&nbsp;When aggregating and merging these massive datasets, I had to be careful of not including future inspections and violations past the permit submit date, along with other data that might bleed into and influence the model. Beyond that, I needed to summarize the datasets on multiple levels and ultimately join them to the permit/address level--both parts took up most of my time on this project. I first joined the permit and datasets on the unique ID (addresskey). After that, I did some data cleaning and removed erroneous segments and weird data.
<p>
 &nbsp;
    </p>

## Feature Engineering
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Improving models using Feature Engineering is sometimes essential. In this case, I manually created new attributes that I believed would be predictive of a building's unsafe status.  In one case, I used a flaggin system, along with data cleanup to create a new variable that indicated when a contractor was handling the construction, or if a homeowner or another entity was responsible.  I also looked at grouping common violation descriptions by highly ranked keywords, and calculated the time between a permit submission and the building's most recent inspection date.
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

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Creating dummy variables vastly increased the number of features in my dataset, which is why selecting important
features and pruning out unimportant features for predicting is necessary.  Two categories that I added dummies to were type of permit work, and permit description, which goes into more detail on what type of construction or building use is needed.  See 
[Categorical Variables](https://gist.github.com/heavenstobetsy/38b48eda46dab9a134b730ebdec7d6c6) for more details.  The next post will go over pruning the features and model selection.

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
