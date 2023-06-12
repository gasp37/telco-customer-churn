from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression, LinearSVC
from pyspark.sql.functions import rand
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


sc = SparkContext(appName='teleco-customer-churn')
spark = SparkSession.builder.getOrCreate()

customers_table = spark.read.csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv', 
                                 header='true', 
                                 inferSchema='true')
customers_table = customers_table.sample(withReplacement=False, fraction=0.15, seed=42)#erase this line for production

def data_wrangling(initial_dataset):
    treated_dataset = initial_dataset.withColumnRenamed('gender', 'Gender')\
                                    .withColumnRenamed('tenure', 'Tenure')\
                                    .withColumnRenamed('customerId', 'CustomerId')
    treated_dataset = treated_dataset.replace(subset='TotalCharges', to_replace=' ', value='0.00')
    treated_dataset = treated_dataset.withColumn('TotalCharges', treated_dataset.TotalCharges.cast('double'))

    return treated_dataset

def transform_string_variables_to_numeric(dataset):
    dataset = dataset.drop('CustomerId')

    string_variables = [variable[0] for variable in dataset.dtypes if variable[1] == 'string']
    string_variables_new_cols = [variable+'_numeric' for variable in string_variables]
    
    indexer_model = StringIndexer(inputCols=string_variables, outputCols=string_variables_new_cols)
    indexer_fitted = indexer_model.fit(dataset)
    dataset_numeric = indexer_fitted.transform(dataset)

    dict_for_renaming_columns = {string_variables_new_cols[index]:string_variables[index] for index in range(len(string_variables))}
    dataset_numeric = dataset_numeric.drop(*string_variables)
    dataset_numeric = dataset_numeric.withColumnsRenamed(dict_for_renaming_columns)

    return dataset_numeric

def vectorize_dataframe(dataframe, label):
    features_cols = dataframe.drop(label).columns

    vecAssembler = VectorAssembler(inputCols=features_cols, outputCol='features')
    vectorized_df = vecAssembler.transform(dataframe)
    vectorized_df = vectorized_df.drop(*features_cols)

    return vectorized_df

def train_test_splitter(dataframe, train_ratio = 0.7, seed=42):
    pre_split_dataframe = dataframe.withColumn('train_test_index', rand(seed=seed))
    
    train_dataframe = pre_split_dataframe.filter(pre_split_dataframe.train_test_index <= train_ratio)
    test_dataframe = pre_split_dataframe.filter(pre_split_dataframe.train_test_index > train_ratio)

    train_dataframe = train_dataframe.drop('train_test_index')
    test_dataframe = test_dataframe.drop('train_test_index')

    print(f'Rows on train dataframe: {train_dataframe.count()}\nRows on test dataframe: {test_dataframe.count()}')
    return train_dataframe, test_dataframe


#----


def evaluate_model(model, dataset, evaluator):
    prediction = model.transform(dataset)

    f1_score = evaluator.evaluate(prediction)
    accuracy_score = evaluator.evaluate(prediction, {evaluator.metricName:'accuracy'})
    recall_score = evaluator.evaluate(prediction, {evaluator.metricName:'recallByLabel'})
    confusion_matrix = prediction.groupBy('Churn', 'prediction').count()
    
    return f1_score, accuracy_score, recall_score, confusion_matrix


def train_lr_model(train_dataset, test_dataset, estimator):
    maxIter_params = estimator['params']['maxIter']
    regParam_params = estimator['params']['regParam']
    lr_model = LogisticRegression(labelCol='Churn')
    evaluator = MulticlassClassificationEvaluator(labelCol='Churn', metricName='f1', metricLabel=1.0)

    
    params_grid = ParamGridBuilder().addGrid(lr_model.maxIter, maxIter_params) \
                                    .addGrid(lr_model.regParam, regParam_params) \
                                    .build()
    
    cv = CrossValidator(estimator=lr_model, estimatorParamMaps=params_grid, evaluator=evaluator)
    cv_fitted = cv.fit(dataset=train_dataset)

    f1_score, accuracy_score, recall_score, confusion_matrix = evaluate_model(cv_fitted.bestModel, test_dataset, evaluator)

    results = {'algorithm':'LogisticRegression',
               'f1_score':f1_score,
               'accuracy':accuracy_score,
               'recall':recall_score,
               'confusion_matrix':confusion_matrix,
               'params':cv_fitted.bestModel.extractParamMap()
               }

    return results

def train_svm_model(train_dataset, test_dataset, estimator):
    maxIter_params = estimator['params']['maxIter']
    regParam_params = estimator['params']['regParam']
    svc_model = LinearSVC(labelCol='Churn')
    evaluator = MulticlassClassificationEvaluator(labelCol='Churn', metricName='f1', metricLabel=1.0)

    
    params_grid = ParamGridBuilder().addGrid(svc_model.maxIter, maxIter_params) \
                                    .addGrid(svc_model.regParam, regParam_params) \
                                    .build()
    
    cv = CrossValidator(estimator=svc_model, estimatorParamMaps=params_grid, evaluator=evaluator)
    cv_fitted = cv.fit(dataset=train_dataset)

    f1_score, accuracy_score, recall_score, confusion_matrix = evaluate_model(cv_fitted.bestModel, test_dataset, evaluator)

    results = {'algorithm':'LogisticRegression',
               'f1_score':f1_score,
               'accuracy':accuracy_score,
               'recall':recall_score,
               'confusion_matrix':confusion_matrix,
               'params':cv_fitted.bestModel.extractParamMap()
               }

    return results

def train_models(train_dataset, test_dataset, estimators:list, evaluator):
    models_results = []
    for estimator in estimators:
        if estimator['algorithm'] == 'LogisticRegression':
            model_results = train_lr_model(train_dataset, test_dataset, estimator)
        if estimator['algorithm'] == 'SVM':
            model_results = train_svm_model(train_dataset, test_dataset, estimator)    
        models_results.append(model_results)
    
    return model_results
    


#---

models_and_params = [{'algorithm':'LogisticRegression', 
                      'params':{'maxIter':[75, 100, 150, 200, 250],
                                'regParam':[100.0, 10.0, 1.0, 0.1, 0.01]
                                }
                    },
                    {'algorithm':'SVM',
                     'params':{'maxIter':[75, 100, 150, 200, 250],
                                'regParam':[100.0, 10.0, 1.0, 0.1, 0.01]
                                }
                    }
                    ]
