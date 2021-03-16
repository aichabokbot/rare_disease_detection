## Spark session creation
from blob_credentials import facts_sas_token, facts_container, workspace_sas_token, workspace_container

from pyspark.sql import SparkSession

myname = "group3"

spark = SparkSession \
    .builder \
    .appName(f"Test-{myname}") \
    .config("spark.executor.instance", "1") \
    .config("spark.executor.memory","512m") \
    .config('spark.jars.packages',"org.apache.hadoop:hadoop-azure:3.1.1") \
    .config("fs.azure", "org.apache.hadoop.fs.azure.NativeAzureFileSystem") \
    .config("fs.wasbs.impl","org.apache.hadoop.fs.azure.NativeAzureFileSystem") \
    .config(f"fs.azure.sas.{facts_container}.hecdf.blob.core.windows.net", facts_sas_token) \
    .config(f"fs.azure.sas.{workspace_container}.hecdf.blob.core.windows.net", workspace_sas_token) \
    .getOrCreate()

## Define your blob services to access files on Azure Blob Storage
from azure.storage.blob import ContainerClient

testname = "koalas-tutorial/datasets/loan_preprocessed.csv"

account_url = "https://hecdf.blob.core.windows.net"

facts_blob_service = ContainerClient(account_url=account_url,
                                     container_name=facts_container,
                                     credential=facts_sas_token)
workspace_blob_service = ContainerClient(account_url=account_url,
                                         container_name=workspace_container,
                                         credential=workspace_sas_token)

        
# Create the parent folder
blobs = list(facts_blob_service.list_blobs())
for blob in blobs:
    from pathlib import Path
    Path(f'../../data/raw/{blob.name}').parent.mkdir(parents=True, exist_ok=True)

    # From facts to your home directory
    with open(f"../../data/raw/{blob.name}", "wb") as data:
        download_stream = facts_blob_service.get_blob_client(blob.name).download_blob()
        data.write(download_stream.readall())

spark.stop()