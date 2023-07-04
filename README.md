# Telephone Company Customer Churn (using PySpark)

In this project, we will analyze a dataframe containing variables related to clients of a telephone company, including social features, contracts, payments, and services. Our goal is to create a classification model that can predict customers likely to churn, enabling us to proactively address and improve customer satisfaction. By identifying potential churners, we can implement strategies to retain them and improve their overall experience.

---

## **Project Steps:**
**Step 0 - Formulate hypotheses**

Formulate hypotheses regarding relations between the many variables and the churn target variable. These hypotheses will be tested and answered throughout the project.

**Step 1 - Analysis:**

Initial Jupyter Notebook with an analysis of data patterns, relations and distribution.

**Step 2 - Models benchmarking:**

Here we'll train a series of models, using different algorithms, hyper-parameters and pre-processing techniques. This will be done using PySpark on a GCP's Dataproc cluster.

This stage includes the following steps: pre-processing data, train different models, evaluate those models with a set of various metrics, and export results for the next step.

**Step 3 - Choose the best model, train it, and export predictions:**

Based on the results exported in step 2, we'll pick the model that best addresses our business problem.

We'll create a script that will train our model using PySpark, and since we don't have new data, we'll use our initial dataset to predict customers likely to churn and assume those are future customers. After that, we can export the data with the predicted values and use it for further analysis.

**Step 4 - Answer hypotheses and create visuals**

Answer the hypotheses that were created in Step 0, and create a dashboard that can answer those questions.

---
## **Step 0 - Formulate hypotheses**

Below you will find a few hypotheses that will be tested and answered:
- The variables in the dataframe hold predictive power in determining whether a customer has churned or not. We'll consider this hypothesis valid if we can create a model with 70% accuracy rate.
- What variables present the strongest influence on a customer leaving the company?
