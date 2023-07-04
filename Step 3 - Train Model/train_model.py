from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LinearSVC
from pyspark.sql.functions import rand, monotonically_increasing_id
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


sc = SparkContext(appName='teleco-customer-churn')
spark = SparkSession.builder.getOrCreate()
sc.setLogLevel('FATAL')
print('Spark version:', sc.version)

raw_customers_table = spark.read.csv('gs://telco-churn-project/data/WA_Fn-UseC_-Telco-Customer-Churn.csv', 
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

    # dict_for_renaming_columns = {string_variables_new_cols[index]:string_variables[index] for index in range(len(string_variables))}
    # dataset_numeric = dataset_numeric.drop(*string_variables)
    # dataset_numeric = rename_columns(dataset_numeric, dict_for_renaming_columns)

    return dataset_numeric, string_variables

def vectorize_dataframe(dataframe, label, string_variables):
    print('-'*36, '\nVectorizing the dataframe')

    features_cols = dataframe.drop(label, *string_variables).columns

    vecAssembler = VectorAssembler(inputCols=features_cols, outputCol='features')
    vectorized_df = vecAssembler.transform(dataframe)
    
    return vectorized_df, features_cols

def train_test_splitter(dataframe, train_ratio = 0.7, seed=42):
    print('-'*36, '\nSplitting train and test tables')
    pre_split_dataframe = dataframe.withColumn('train_test_index', rand(seed=seed))
    
    train_dataframe = pre_split_dataframe.filter(pre_split_dataframe.train_test_index <= train_ratio)
    test_dataframe = pre_split_dataframe.filter(pre_split_dataframe.train_test_index > train_ratio)

    train_dataframe = train_dataframe.drop('train_test_index')
    test_dataframe = test_dataframe.drop('train_test_index')

    print(f'Rows on train dataframe: {train_dataframe.count()}\nRows on test dataframe: {test_dataframe.count()}')
    return train_dataframe, test_dataframe

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

def train_svm_model(train_dataset, test_dataset, params, label):
    maxIter_params = params['maxIter']
    regParam_params = params['regParam']
    svc_model = LinearSVC(featuresCol='features', labelCol=label)
    evaluator = MulticlassClassificationEvaluator(labelCol=label, metricName='f1', metricLabel=1.0)

    
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
               'confusion_matrix':confusion_matrix
               }

    return cv_fitted, results

params = {
        'maxIter':[250],
        'regParam':[0.001]
        }

customers_table = data_wrangling(raw_customers_table)
customers_table, string_variables = transform_string_variables_to_numeric(customers_table)
customers_table, features_cols = vectorize_dataframe(customers_table, 'Churn_numeric', string_variables)
train_table, test_table = train_test_splitter(customers_table)

svm_model, results = train_svm_model(train_table, test_table, params, label='Churn_numeric')

transformed = svm_model.transform(customers_table)
customers_table_with_prediction = transformed.select(*string_variables, 'SeniorCitizen', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'prediction')

customers_table_with_prediction.coalesce(1).write.mode('overwrite').option('header',True).csv('gs://telco-churn-project/results/predicted_dataset')

print('-'*36, '\nEncerrando a Spark Session e o Spark Context...')
spark.stop()
sc.stop()