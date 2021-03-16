<p float="center">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/philly-skyline-extended.jpg" width="550" />
</p>

### Motivation

&nbsp;&nbsp;&nbsp;&nbsp;Philadelphia is a unique city. At its peak in 1960, Philadelphia was home to over two million people in 1950; due to deindustrialization starting in the 1960s, along with numerous other factors, the population shrunk over the next four decades, crashing its housing market. 

&nbsp;&nbsp;&nbsp;&nbsp;In the last seven years, however, Philadelphia home prices have risen almost 46%, although the consequence of its previous population decline means much of the city's housing stock is in bad shape. As a result, there are a large number of buildings which fail inspections, in addition to the constant new construction and renovation of a city undergoing a revival - all of which fall under Philadelphia's Department of Licenses and Inspections (L&I).



![Bar chart of permit reason](https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/permits.png)
[Fig. 1 – Bar Chart of 2019 Ranked Building Permits]

<p>
	
&nbsp;&nbsp;&nbsp;&nbsp;I developed this dataset and project for several reasons, one of which was the complexity and messiness of the data.  This project required extensive cleaning, aggregating, and pre-processing of several datasets before it was useable. Not only is it a real-life problem that can be solved, but there was no guide or pre-constructed and cleaned dataset that thousands of people have already run through. 

&nbsp;&nbsp;&nbsp;&nbsp;In real life, data is often messy and ugly--even in the tech world! I firmly believe that one of the best skills a data scientist can develop is the ability to turn a vague idea and messy dataset into a solvable problem.
	
&nbsp;
&nbsp;
	</p>
	
### Objective

&nbsp;&nbsp;&nbsp;&nbsp;My aim was to create a robust model which actively uses the Philadelphia Licenses and Inspections (L&I) permit application process as a monitoring system for potentially dangerous buildings. 

&nbsp;&nbsp;&nbsp;&nbsp;This model combines two aspects: the building permit process, and the building's historical inspection, violation, and permit data. Most egregious code violations are currently found by either L&I inspectors making the rounds, or complaints stemming from people's observations of a building's exterior. However, not many people who have access to a building's interior - whether it's the owner, contractor, or tenant - have the incentive or the structural knowledge to realize and complain about dangerous conditions or fire hazards. By using the permit process, we may be able to find unsafe buildings that have been overlooked by the usual system. 

&nbsp;&nbsp;&nbsp;&nbsp;How the model works: When fed a new permit application, the model predicts how dangerous or unsafe the building could be, and whether L&I inspectors should immediately follow up. By predicting which buildings have a higher likelihood of being condemned and what buildings are the most dangerous, L&I can prioritize the buildings that inspectors should focus on first - and by doing that, increase the safety of Philadelphians. One doesn't have to look far in the news to find Philadelphia's unsafe housing risk, which is why a streamlined and
[accurate inspection process is crucial.](https://whyy.org/segments/renter-beware-phillys-deadly-housing-problem/)

<p>

&nbsp;
&nbsp;
	</p>
### The Permit Process

&nbsp;&nbsp;&nbsp;&nbsp;In 2019, there were over 14,000 permits issued for building and renovation in Philadelphia. These permits range from electrical work being done by contractors, to minor alterations done by homeowners, to demolitions.  The large volume of permits means that L&I can't possibly send inspectors to check on every permit or inspection request in a timely manner.  

&nbsp;&nbsp;&nbsp;&nbsp;A large number of buildings in Philadelphia fail their inspections and further followups, while some buildings have more drastic action being taken, and are declared unsafe and uninhabitable. 

<p float="center">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/Inspection_outcomes.png" width="300" />
  <img src="https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/building_failures.png" width="440" /> 
</p>
[Figs. 2&3 – Donut Charts of 2019 Inspection Outcomes and Repeat Failures]

<p>

&nbsp;
&nbsp;
	</p>
	
### Modeling Process

&nbsp;&nbsp;&nbsp;&nbsp;I built this model using data from Philadelphia's open source data repository: [OpenDataPhilly](https://www.opendataphilly.org/). Using the city's APIs, I pulled in four large datasets: permits data, inspection data, code violation data, and unsafe violations data. I then cleaned, aggregated, and blended the data together before building the model.

&nbsp;&nbsp;&nbsp;&nbsp;In addition to the variables and aggregations from previous permits, inspections, and violations were included in the dataset, I also built custom variables in order to improve the model's prediction accuracy.

&nbsp;&nbsp;&nbsp;&nbsp;When I started comparing the unsafe buildings vs "safe" buildings, the dataset was clearly unbalanced - due to the sheer volume of permits coming through the system, <1% of buildings were unsafe. Blindly throwing this data into a model might result in a high accuracy rate- of course it's easy to guess a building is safe when 99% of the time you'll be correct. However, the model will be terrible when actually trying to find the unsafe buildings.

&nbsp;&nbsp;&nbsp;&nbsp;In order to correct for this, I spent some time finessing the dataset further by resampling the data. I looked at both undersampling and SMOTE in order to re-balance the data.

&nbsp;&nbsp;&nbsp;&nbsp;I ended up pulling a large number of features, as I wasn't sure which ones would later play an important role: therefore finding variable importance was key in slimming down and improving the model.  I used Boruta for feature selection, which was used to improve model quality. Below is a graph of the most important variables used in the model.

![Variable Importance](https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/most_feature_importanes.png)
</p>
[Fig. 4 – Scaled variable importance among predictors]
</p>

<p>

&nbsp;&nbsp;&nbsp;&nbsp;When designing the actual model, I reviewed a variety of models; I then used ensemble learning to see if combining models would improve the results. For more details on the modeling process, see my [Github](https://github.com/heavenstobetsy/PhillyConstruction), in addition to an upcoming post.

<p>

&nbsp;
&nbsp;
	</p>

### Results

Baseline Accuracy

![Confusion Matrix chart](https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/confusion_matrix.png)
</p>
[Fig. 5 – Confusion matrix, breaking out model accuracy]
</p>

<p>

&nbsp;
&nbsp;
	</p>


	
### Conclusion

&nbsp;&nbsp;&nbsp;&nbsp;My work isn't done yet--I plan on adjusting the model and trying to improve its accuracy even more by improving upon the ensemle model.  I believe that open access to government data is essential, and can spur changes in government policy and methodology.  

&nbsp;&nbsp;&nbsp;&nbsp;Finally, although this model tackles a small, yet serious problem, the dataset and model that I built can be tweaked for a wide array of uses. What about predicting buildings that will fail inspections multiple times, or finding neighborhood trends before they're finished? Or maybe recommend successful contractors to permit applicants based on their past successes, and monitor contractors who repeatedly incur code violations?

&nbsp;
&nbsp;
&nbsp;
	

### Further Reading

&nbsp;&nbsp;&nbsp;&nbsp;[Philadelphia's Deadly Housing Problem](https://whyy.org/segments/renter-beware-phillys-deadly-housing-problem/)

&nbsp;&nbsp;&nbsp;&nbsp;[State of Philadelphia's Housing Market](https://www.inquirer.com/real-estate/housing/signs-of-recession-philadelphia-suburbs-housing-market-real-estate-prices-hot-20190916.html)


