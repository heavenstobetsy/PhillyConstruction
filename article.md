### Objective

&nbsp;&nbsp;&nbsp;&nbsp;My objective is to create a robust model that when fed a L&I permit application, then predicts how dangerous or unsafe the building could be, and whether L&I inspectors should immediately follow up. By predicting which buildings have a higher liklihood of being condemned, and what buildings are the most dangerous, L&I can prioritize the buildings that inspectors should focus on first, and by doing that, increase the safety of Philadelphians.  One doesn't have to look far in the news to find the [the unfortunate results of unsafe housing.](https://whyy.org/segments/renter-beware-phillys-deadly-housing-problem/)



### Motivation

&nbsp;&nbsp;&nbsp;&nbsp;Philadelphia is a unique city. At its peak in 1960, Philadelphia was home to over two million people in 1950; due deindustrialization starting in the 1960s, along with numerous other factors, the population shrunk over the next four decades, crashing its housing market.  In the last seven years, however, Philadelphia home prices have risen almost 46%, although the consequence of its previous population decline means much of the city’s housing stock is in bad shape.  As a result, there are a large number of buildings which fail inspections, in addition to the constant new construction and renovation of a city undergoing a revival--all of which fall under Philadelphia's Department of Licenses and Inspections (L&I).

&nbsp;&nbsp;&nbsp;&nbsp;In 2019, there were over 14,000 permits issued for building and renovation in Philadelphia. These permits range from electrical work being done by contractors, to minor alterations done by homeowners, to demolitions.  The large volume of permits means that L&I can't possibly send inspectors to check on every permit or inspection request in a timely manner.  A not insignificant number of buildings in Philadelphia fail their inspections and further followups, with some buildings being declared unsafe and uninhabitable. 

Before diving into the methodology, let’s consider the benefits of such a model if it were to be implemented. Imagine a large city with lots of construction and high rates of buildings failing to comply with the local code. The department tasked with ensuring that buildings comply doesn’t have nearly the capacity to send inspectors to every building in the city. The department allocates initial inspections based on number of events and criteria. A number of these buildings go on to fail their inspections. Once they have failed, they must be re-inspected later to ensure that each makes the prescribed improvements. Some of these re-inspections are on a fixed schedule and some aren’t. Either way, the department has limited capacity. They aren’t able to keep up with all of the existing cases and also inspect new and potentially unsafe buildings.

This type of predictive model would provide the person in charge of allocating inspectors with the probabilities of which buildings are most likely to remain non compliant. She could combine these probabilities with other internal decision factors and plan accordingly. By prioritizing inspections for buildings that are likely to fail, she can ensure that these problematic cases comply in a reasonable amount of time. Meanwhile she can wait to check in on buildings that are likely to pass inspection anyway. All the while this may free up additional inspectors to find undiscovered potentially dangerous buildings using community feedback, professional intuition, or an additional predictive model. When all is said and done, this analysis could allow the inspectors to create safer buildings all across the city.
### Modeling Process

### Results

### Further Reading
https://whyy.org/segments/renter-beware-phillys-deadly-housing-problem/
https://www.inquirer.com/real-estate/housing/signs-of-recession-philadelphia-suburbs-housing-market-real-estate-prices-hot-20190916.html

### Conclusion
