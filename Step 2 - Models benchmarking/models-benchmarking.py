from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression, LinearSVC, RandomForestClassifier
from pyspark.sql.functions import rand
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


sc = SparkContext(appName='teleco-customer-churn')
spark = SparkSession.builder.getOrCreate()
sc.setLogLevel('FATAL')
print('Spark version:', sc.version)

customers_table = spark.read.csv('gs://telco-churn-project/data/WA_Fn-UseC_-Telco-Customer-Churn.csv', 
                                 header='true', 
                                 inferSchema='true')


def data_wrangling(initial_dataset):
    print('-'*36, '\nStarting Data Wrangling')
    treated_dataset = initial_dataset.withColumnRenamed('gender', 'Gender')\
                                    .withColumnRenamed('tenure', 'Tenure')\
                                    .withColumnRenamed('customerId', 'CustomerId')
    treated_dataset = treated_dataset.replace(subset='TotalCharges', to_replace=' ', value='0.00')
    treated_dataset = treated_dataset.withColumn('TotalCharges', treated_dataset.TotalCharges.cast('double'))

    return treated_dataset

# In spark versions earlier than 3.4.0 we can use the method DataFrame.withColumnsRenamed().
# Since Dataproc currently only has up to the 3.3.0 version available, we'll create this simple function.
def rename_columns(dataset, dict):
    for column in dict:
        dataset = dataset.withColumnRenamed(column, dict[column])
    return dataset
    

def transform_string_variables_to_numeric(dataset):
    print('-'*36, '\nTranforming string variables to numeric variables')

    dataset = dataset.drop('CustomerId')

    string_variables = [variable[0] for variable in dataset.dtypes if variable[1] == 'string']
    string_variables_new_cols = [variable+'_numeric' for variable in string_variables]
    
    indexer_model = StringIndexer(inputCols=string_variables, outputCols=string_variables_new_cols)
    indexer_fitted = indexer_model.fit(dataset)
    dataset_numeric = indexer_fitted.transform(dataset)

    dict_for_renaming_columns = {string_variables_new_cols[index]:string_variables[index] for index in range(len(string_variables))}
    dataset_numeric = dataset_numeric.drop(*string_variables)
    dataset_numeric = rename_columns(dataset_numeric, dict_for_renaming_columns)

    return dataset_numeric

def vectorize_dataframe(dataframe, label):
    print('-'*36, '\nVectorizing the dataframe')

    features_cols = dataframe.drop(label).columns

    vecAssembler = VectorAssembler(inputCols=features_cols, outputCol='features')
    vectorized_df = vecAssembler.transform(dataframe)
    vectorized_df = vectorized_df.drop(*features_cols)

    return vectorized_df

def train_test_splitter(dataframe, train_ratio = 0.7, seed=42):
    print('-'*36, '\nSplitting train and test tables')
    pre_split_dataframe = dataframe.withColumn('train_test_index', rand(seed=seed))
    
    train_dataframe = pre_split_dataframe.filter(pre_split_dataframe.train_test_index <= train_ratio)
    test_dataframe = pre_split_dataframe.filter(pre_split_dataframe.train_test_index > train_ratio)

    train_dataframe = train_dataframe.drop('train_test_index')
    test_dataframe = test_dataframe.drop('train_test_index')

    print(f'Rows on train dataframe: {train_dataframe.count()}\nRows on test dataframe: {test_dataframe.count()}')
    return train_dataframe, test_dataframe


#----


def undersample(dataset, sample=0.4):
    undersampled_label = dataset.filter('Churn == 0').sample(sample)
    undersampled_dataset = undersampled_label.union(dataset.filter('Churn == 1'))
    print('Undersampled dataset label ratio:\n')
    undersampled_dataset.groupBy('Churn').count().show()

    return undersampled_dataset

def evaluate_model(model, dataset, evaluator):
    prediction = model.transform(dataset)

    f1_score = evaluator.evaluate(prediction)
    accuracy_score = evaluator.evaluate(prediction, {evaluator.metricName:'accuracy'})
    recall_score = evaluator.evaluate(prediction, {evaluator.metricName:'recallByLabel'})
    confusion_matrix = prediction.groupBy('Churn', 'prediction').count().collect()
    
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
               'params':str(cv_fitted.bestModel.extractParamMap())
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

    results = {'algorithm':'SVM',
               'f1_score':f1_score,
               'accuracy':accuracy_score,
               'recall':recall_score,
               'confusion_matrix':confusion_matrix,
               'params':str(cv_fitted.bestModel.extractParamMap())
               }

    return results

def train_random_forest_model(train_dataset, test_dataset, estimator):
    print('Training Random Forest')
    featureSubsetStrategy_params = estimator['params']['featureSubsetStrategy']
    numTrees_params = estimator['params']['numTrees']
    random_forest_model = RandomForestClassifier(labelCol='Churn')
    evaluator = MulticlassClassificationEvaluator(labelCol='Churn', metricName='f1', metricLabel=1.0)

    
    params_grid = ParamGridBuilder().addGrid(random_forest_model.featureSubsetStrategy, featureSubsetStrategy_params) \
                                    .addGrid(random_forest_model.numTrees, numTrees_params) \
                                    .build()
    
    cv = CrossValidator(estimator=random_forest_model, estimatorParamMaps=params_grid, evaluator=evaluator)
    cv_fitted = cv.fit(dataset=train_dataset)

    f1_score, accuracy_score, recall_score, confusion_matrix = evaluate_model(cv_fitted.bestModel, test_dataset, evaluator)

    results = {'algorithm':'RandomForest',
               'f1_score':f1_score,
               'accuracy':accuracy_score,
               'recall':recall_score,
               'confusion_matrix':confusion_matrix,
               'params':str(cv_fitted.bestModel.extractParamMap())
               }

    return results

def train_models(train_dataset, test_dataset, estimators:list):
    models_results = []
    for estimator in estimators:
        local_train_dataset = train_dataset

        if estimator['undersample'] == True:
            local_train_dataset = undersample(train_dataset)

        if estimator['algorithm'] == 'LogisticRegression':
            print('-'*36, '\nTraining Logistic Regression Model')
            model_results = train_lr_model(local_train_dataset, test_dataset, estimator)
        if estimator['algorithm'] == 'SVM':
            print('-'*36, '\nTraining SVM Model')
            model_results = train_svm_model(local_train_dataset, test_dataset, estimator)
        if estimator['algorithm'] == 'RandomForest':
            print('-'*36, '\nTraining Random Forest Model')
            model_results = train_random_forest_model(local_train_dataset, test_dataset, estimator)

        models_results.append(model_results)
    
    return models_results
    


#---

models_and_params = [{'algorithm':'LogisticRegression',
                      'undersample':False, 
                      'params':{'maxIter':[75, 100, 150, 200, 250],
                                'regParam':[100, 10., 1.0, 0.1, 0.01, 0.001]
                                }
                    },
                    {'algorithm':'LogisticRegression',
                      'undersample':True, 
                      'params':{'maxIter':[75, 100, 150, 200, 250],
                                'regParam':[100, 10., 1.0, 0.1, 0.01, 0.001]
                                }
                    },
                    {'algorithm':'SVM',
                     'undersample':True,
                     'params':{'maxIter':[75, 100, 150, 200, 250],
                                'regParam':[100, 10., 1.0, 0.1, 0.01, 0.001]
                                }
                    },
                    {'algorithm':'RandomForest',
                     'undersample':True,
                     'params':{'featureSubsetStrategy':['1', '2', '4', '6', '9'],
                               'numTrees':[10, 20, 100],
                               'maxDepth':[3, 5, 7, 9, 11],
                               'minInstancesPerNode':[1, 2, 3, 4, 5],
                               'impurity':['gini', 'entropy']
                         
                     }
                    }
                    ]


customers_table = data_wrangling(customers_table)
customers_table = transform_string_variables_to_numeric(customers_table)
customers_table = vectorize_dataframe(customers_table, 'Churn')
train_table, test_table = train_test_splitter(customers_table)

results = train_models(train_dataset=train_table, test_dataset=test_table, estimators=models_and_params)


print('-'*36, '\nSalvando os resultados...')
sdf_results = spark.createDataFrame(results)
sdf_results.coalesce(1).write.mode('overwrite').format('json').save('gs://telco-churn-project/results')


print('-'*36, '\nEncerrando a Spark Session e o Spark Context...')
spark.stop()
sc.stop()