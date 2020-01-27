
### Objective

&nbsp;&nbsp;&nbsp;&nbsp;My objective is to create a robust model which actively uses the Philadelphia Licenses and Inspections (L&I) permit application process as a monitoring system for potentially dangerous buildings. When fed the permit application, the model then predicts how dangerous or unsafe the building could be, and whether L&I inspectors should immediately follow up. By predicting which buildings have a higher likelihood of being condemned and what buildings are the most dangerous, L&I can prioritize the buildings that inspectors should focus on first--and by doing that, increase the safety of Philadelphians.  One doesn't have to look far in the news to find [Philadelphia's unsafe housing risk.](https://whyy.org/segments/renter-beware-phillys-deadly-housing-problem/)



### Motivation

&nbsp;&nbsp;&nbsp;&nbsp;Philadelphia is a unique city. At its peak in 1960, Philadelphia was home to over two million people in 1950; due deindustrialization starting in the 1960s, along with numerous other factors, the population shrunk over the next four decades, crashing its housing market.  In the last seven years, however, Philadelphia home prices have risen almost 46%, although the consequence of its previous population decline means much of the city’s housing stock is in bad shape.  As a result, there are a large number of buildings which fail inspections, in addition to the constant new construction and renovation of a city undergoing a revival--all of which fall under Philadelphia's Department of Licenses and Inspections (L&I).


![Bar chart of permit reason](https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/permits.png)
[Fig. 1 – Bar Chart of 2019 Ranked Building Permits]

&nbsp;
&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;In 2019, there were over 14,000 permits issued for building and renovation in Philadelphia. These permits range from electrical work being done by contractors, to minor alterations done by homeowners, to demolitions.  The large volume of permits means that L&I can't possibly send inspectors to check on every permit or inspection request in a timely manner.  A not insignificant number of buildings in Philadelphia fail their inspections and further followups, with some buildings being declared unsafe and uninhabitable. 

<p float="center">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/Inspection_outcomes.png" width="300" />
  <img src="https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/building_failures.png" width="400" /> 
</p>
[Figs. 2&3 – Donut Charts of 2019 Inspection Outcomes and Repeat Failures]

### Modeling Process

&nbsp;&nbsp;&nbsp;&nbsp;I built this model using data from Philadelphia's open source data repository: [OpenDataPhilly](https://www.opendataphilly.org/). Using the city's APIs, I pulled in four large datasets: permits data, inspection data, code violation data, and unsafe violations data, which I then cleaned, aggregated, and blended together before building the model. Features from permits, inspections, and violations are used in the dataset in order to create a prediction model. The model was finessed further using SMOTE oversampling for balancing the data, and Boruta for feature selection to improve model quality. The modeling process is gone over in more at my [Github](https://github.com/heavenstobetsy/PhillyConstruction), in addition to an upcoming post.


### Results

Baseline Accuracy

![Confusion Matrix chart](https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/confusion_matrix.png)
</p>
[Fig. 4 – Confusion matrix, breaking out model accuracy]
</p>

### Variable Importance 
&nbsp;&nbsp;&nbsp;&nbsp;I ended up pulling a large number of features, as I wasn't sure which ones would later play an important role: therefore finding variable importance was key in slimming down and improving the model.  Below is a graph of the most important variables used in the model.

![Scaled variable importance](https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/feature_importsance.png)
<p>
[Fig. 5 – Scaled variable importance among predictor variables]


### Conclusion

&nbsp;&nbsp;&nbsp;&nbsp;My work isn't done yet--I plan on adjusting the model and trying to improve its accuracy even more.  I believe that open access to government data is essential, and can spur changes in government policy and methodology.  &nbsp;&nbsp;&nbsp;&nbsp;Finally, although this model tackles a small, yet serious problem, the dataset and model that I built can be tweaked for a wide array of uses. What about predicting buildings that will fail inspections multiple times, or finding neighborhood trends before they're finished? Or maybe recommend successful contractors to permit applicants based on their past successes, and monitor contractors who repeatedly incur code violations?

&nbsp;
&nbsp;
&nbsp;
	

### Further Reading

&nbsp;&nbsp;&nbsp;&nbsp;[Philadelphia's Deadly Housing Problem](https://whyy.org/segments/renter-beware-phillys-deadly-housing-problem/)

&nbsp;&nbsp;&nbsp;&nbsp;[State of Philadelphia's Housing Market](https://www.inquirer.com/real-estate/housing/signs-of-recession-philadelphia-suburbs-housing-market-real-estate-prices-hot-20190916.html)


