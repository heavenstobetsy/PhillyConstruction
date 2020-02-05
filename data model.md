
## Imbalanced Data

&nbsp;&nbsp;&nbsp;&nbsp;After cleaning and blending the data, I added in a target to indicate what addresses received an unsafe violation after their permit submit date.

![Adding in a target](https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/target.png)

Looking at the shape of the dataset, it's clear that it's very imbalanced - 99% of the data does not have an unsafe violation. There are a few different techniques for dealing with imbalanced dataset - I looked at two for this project: undersampling the majority class (safe buildings) and SMOTE.

<p>
 &nbsp;
    </p>
    
## Undersampling/SMOTE

&nbsp;&nbsp;&nbsp;&nbsp;*Undersampling:* is the process where you randomly delete some of the observations from the majority class (safe buildings) in order to match the the minority class (unsafe buildings).

![Undersampling](https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/undersampling.png)

&nbsp;&nbsp;&nbsp;&nbsp;*SMOTE*: is short for Synthetic Minority Over-sampling Technique. Basically, it looks at the feature space for the minority class data points and considers its k nearest neighbors.

![SMOTE](https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/smote_gist.png)

&nbsp;&nbsp;&nbsp;&nbsp;The scatter charts below show how SMOTE resamples data to make new data points. The blue points belong to the majority class, while the orange points belong to the minority class. In the second chart, you can see the additional data that has been generated for the minority class.




<p>
 &nbsp;
    </p>
    
## Feature Selection with Boruta
&nbsp;&nbsp;&nbsp;&nbsp;
<p>
 &nbsp;
    </p>

## Choosing a Model
&nbsp;&nbsp;&nbsp;&nbsp;

<p>
 &nbsp;
    </p>

## Model Types
&nbsp;&nbsp;&nbsp;&nbsp;

<p>
 &nbsp;
    </p>
 
 ## Hyperparameter Optimization 
&nbsp;&nbsp;&nbsp;&nbsp;

<p>
 &nbsp;
    </p>
    
    
 ## Ensemble Learning
&nbsp;&nbsp;&nbsp;&nbsp;

<p>
 &nbsp;
    </p>
    
