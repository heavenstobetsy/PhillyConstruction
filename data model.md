
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

![Pre-SMOTE](https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/pre_smote.png)
![SMOTE](https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/smote.png)


&nbsp;&nbsp;&nbsp;&nbsp;Ultimately, I found that 1) Feature selection with SMOTE takes a lot of time and 2) while the precision is high using SMOTE, recall is poor. So why did SMOTE perform so poorly in this example? [This paper](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-14-106) suggests that SMOTE performs poorly when given high-dimensional class-imbalanced data - which is exactly the type of data that we have.

<p>
 &nbsp;
    </p>
    
    
## Splitting the Data, Pre-processing, then Feature Selection with Boruta

&nbsp;&nbsp;&nbsp;&nbsp;Before pruning unneeded features with Boruta, I randomly split the data into test and training sets, with 70% of the data assigned to a training set and 30% assigned to the test set.  
&nbsp;&nbsp;&nbsp;&nbsp;In addition, I standardized the data using StandardScaler from sklearn.   StandardScaler transforms my dataset, so that each features' distribution will have a mean value 0 and standard deviation of 1. 


![Test train split](https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/test_train.png)

&nbsp;&nbsp;&nbsp;&nbsp;When building a model, it can be hard it is to identify which features are important and which are just noise - useless data which can interfere with training. Increasingly, high-dimensional data has become the new normal and pruning features has become a necessity. Removing noisy features will help reduce training time and improve the accuracy of a model - in addition, removing features will help avoid  model overfitting.

&nbsp;&nbsp;&nbsp;&nbsp;To prune features, I used Boruta, which is package available for both python and R. Boruta is a wrapper built around the random forest classification algorithm, which assists in capturing potentially valuable features in the dataset with respect to the outcome variable. Boruta then goes through multiple iterations to determine whether a feature is worth keeping.

![Boruta part I](https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/boruta_gist1.png)

![Boruta part II](https://github.com/heavenstobetsy/PhillyConstruction/blob/master/Charts/boruta_gist2.png)


<p>
 &nbsp;
    </p>


## Choosing a Model: Model Types, Ensemble Learning, & Hyperparameter Optimization 

&nbsp;&nbsp;&nbsp;&nbsp;I used ROC AUC/AUROC as one of the main criteria for determining the best model. AUROC is a good choice for a classifier problem, as it gives a sense of how well the classifier can be tuned to be more or less sensitive - and can also get the best outcomes by changing the class threshold.

![AUC ROC Scores](https://raw.githubusercontent.com/heavenstobetsy/PhillyConstruction/master/Charts/Model%20Scores%20(ROC-AUC).png)

&nbsp;&nbsp;&nbsp;&nbsp;While trying to find the best model, I tried out a variety of different types of models:  from the most basic model - logistic regression - to XGBoost - to ensemble learning. For the initial review, the XGBClassifier performed the best with a score of 0.919. Translating the model to actual predictions results in the predicions below.

![XGB Initial Results](https://raw.githubusercontent.com/heavenstobetsy/PhillyConstruction/master/Charts/XGB%20Initial%20Results.png)


&nbsp;&nbsp;&nbsp;&nbsp;However, I wanted to try to see if ensemble learning could perform better. Ensemble learning combines the predictions from multiple models to reduce the variance of predictions and reduce generalization error. However, for an ensemble strategy to work, prediction errors must be relatively uncorrelated.

&nbsp;&nbsp;&nbsp;&nbsp;Based on the correlation matrix below, scores are low which shows promise for the ensemble model.  The matrix looks at error correlations on a class prediction basis--whether a building is scored unsafe or not.

![Correlation Matrix](https://raw.githubusercontent.com/heavenstobetsy/PhillyConstruction/master/Charts/Correlation_matrix.png)

&nbsp;&nbsp;&nbsp;&nbsp;The two charts below show how each model performs separately, and how they rank against each other.
![Ensemble ROC Curve](https://raw.githubusercontent.com/heavenstobetsy/PhillyConstruction/master/Charts/Emsemble_ROC_Curve.png)
![Model Performace Rank](https://raw.githubusercontent.com/heavenstobetsy/PhillyConstruction/master/Charts/model_performance_rank.png)

&nbsp;&nbsp;&nbsp;&nbsp;After pruning three of the lower performing models, the final score is 0.914--so almost as good as the XGBClassifier model.
![Final Ensemble Score](https://raw.githubusercontent.com/heavenstobetsy/PhillyConstruction/master/Charts/Final_Ensemble_Score.png)
