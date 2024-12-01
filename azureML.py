#import required libraries
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

#Enter details of your Azure Machine Learning workspace
subscription_id = "caa7bcdb-c3e1-4687-a73b-7b621b7e4b23"
resource_group = "rezolve-ai"
workspace = "ft-workspace"

#connect to the workspace
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)


computes = ml_client.compute.list()
for compute in computes:
    print(compute.name)