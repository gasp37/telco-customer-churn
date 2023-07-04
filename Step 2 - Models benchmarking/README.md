# Step 2 - Models benchmarking

In this step we'll train our many models using different algorithms and hyper-parameters, we'll also use cross-validation and undersampling to deal with our unbalanced dataset, and grid-search to test the different hyper-parameters. After we train those models, we will use a few metrics to evaluate their performances and compare it against each other.

In this folder you'll find the following files:
- gcp_shell_commands.txt - The commands we'll use on our GCP shell to create our Dataproc cluster, submit a job that will run our PySpark script, and delete the cluster after it's all done.
- models-benchmarking.py - Our PySpark script containing all the steps to create our models. From pre-processing, to training, to testing/evaluating and finally export those results.
- results_from_dataproc.json - The results from our benchmarking script. Here you can find the metrics and parameters for each model trained.