
## Building the Model
&nbsp;&nbsp;&nbsp;&nbsp;This model is meant to act as a tool for Philadelphia’s Department of Licenses and Inspections to 
prioritize which buildings to investigate after a building permit is submitted. Specifically, I wanted to figure out which 
building would be most likely be declared unsafe and extremely hazardess and needs immediate action taken.  For a less 
technical summary, along with my motivation and reasoning for choosing this problem see in my Medium write-up
at [link].

## Datasets
&nbsp;&nbsp;&nbsp;&nbsp;For this model I used data from The City of Philadelphia’s Department of Licenses and Inspections. 
To keep the data manageable, I focused on buildings with a permit submitted up to through June 2019. The L&I data was
built into a dataframe comprised of several datasets including permits, inspections, code violations, and unsafe violations. 
This includes:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Permits: data on building construction/use/update permits. <p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Inspectiosn: data on cases and compliance/non-compliance follow-up. <p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Violations: data and descriptions of every building code violation, and risk level. <p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Unsafe Violations: data and descriptions of the unsafe building code violations. <p>
&nbsp;
&nbsp;
<p>
<p>
This chart shows the breakdown of the total number of buildings for each type of data:



## Data Wrangling

## Feature Engineering

## Categorical Variables

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Creating dummy variables vastly increased the number of features in my dataset, which is why selecting important
features and pruning out unimportant features for predicting is necessary.  The next post will go over pruning the features and model selection.

<p>
<p>
<p>
<p>
<p>
<p>
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This project was completed by E. Johnson, and last updated in January 2020. The data I used came from the 
City of Philadelphia Departments of Licenses and Inspections located, at [OpenDataPhilly](https://www.opendataphilly.org/). I conducted the analysis using 
Python and used the following packages: AWS SageMaker, pandas,numpy, sklearn, matplotlib, imbalanced, xgboost, Boruta, seaborn, and waffle.
