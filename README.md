Project developed by: [Gabriel Pacheco](https://linkedin.com/in/gabriel-pacheco37)

---
# Improving a telephone company customer churn rate (using PySpark)
In this project, we will analyze a dataframe containing variables related to clients of a telephone company, including social features, contracts, payments, and services. Our two main goals are to:
- Create a classification model that can predict customers likely to churn.
- Based on the data, define strategies to proactively address and retain those customers that are at risk.
---
This project starts from the following hypothesis, and draw the following conclusion:

**Hypothesis:**  It's possible to predict if a customer will leave the company. We'll consider this hypothesis valid if we can create a model with 70% recall rate.

**Conclusion:**  Yes. Using a SVC model with a 77% recall value, and a 76% F1-score.

## **Project Steps:**
**Step 0 - Understanding the business problem**

To understand better how to handle this business problem, we can cross it with a confusion matrix that represents each possible outcome of our model. This makes it easier to comprehend what each outcome represents for the company.

*True Positive*: We successfully predicted a churn and were able to take action and try to prevent it.

*False Positive*: We wrongfully predicted a churn, we took action but it had no effect, since the customer wasn't going to leave.

*True Negative*: We successfully predicted a no-churn, and didn't need to act.

*False Negative*: We wrongfully predicted a no-churn, no action was took and we lost the client. ***Worst-case scenario***

So for our model to get the best results, we need to minimize the False Negatives and maximize the True Positives and True Negatives. For this, we can use both our Recall metric and F1-Score, which are great to measure False Negatives. 

**Step 1 - Initial analysis:**

Initial Jupyter Notebook with an analysis of data patterns, relations and distribution.

**Step 2 - Models benchmarking:**

Here we train a series of models, using different algorithms, hyper-parameters and pre-processing techniques. This is done using PySpark on a GCP's Dataproc cluster.

This stage includes the following steps: pre-processing data, train different models, evaluate those models with a set of various metrics, and export results for the next step.

From this step we extracted the following table:

|Model|Accuracy|F1-Score|Recall|
|---|---|---|---|
|Logistic Regression - Without Undersample|**0.807**|**0.798**|0.521|
|Logistic Regression|0.759|0.769|0.731|
|**SVM**|0.753|0.765|**0.770**|
|Random Forest|**0.762**|**0.772**|0.753|

Using the recall and f1-score metrics as a basis, we can establish our SVM/SVC model, as our best model. 

**Step 3 - Choose the best model, train it, and export predictions:**

Based on the results exported in step 2, we'll pick the model that best addresses our business problem.

We'll create a script that will train our model using PySpark, and since we don't have new data, we'll use our initial dataset to predict customers likely to churn and assume those as future customers. After that, we can export the data with the predicted values and use it for further analysis.

**Step 4 - Draw strategies**

With our model and it's predictions, we can analyze our data again, but now, with the purpose of finding patterns and draw strategies to handle the possible churn cases.

