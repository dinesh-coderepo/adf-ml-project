import tensorflow as tf
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import numpy as np
import io
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Download MNIST data and upload to Azure Blob storage.')
parser.add_argument('--storage-account', required=True, help='Azure Storage account name')
parser.add_argument('--container', required=True, help='Blob container name')
parser.add_argument('--job-id', help='Batch job ID (optional)')
args = parser.parse_args()

# Azure Storage account details
storage_account_name = args.storage_account
container_name = args.container
job_id = args.job_id

# Download MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Initialize the BlobServiceClient using the default credential
credential = DefaultAzureCredential()
account_url = f"https://{storage_account_name}.blob.core.windows.net"
blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)

# Create a container (if it doesn't exist)
container_client = blob_service_client.get_container_client(container_name)
if not container_client.exists():
    container_client.create_container()

# Function to save numpy array as blob
def save_array_to_blob(array, name):
    bytes_io = io.BytesIO()
    np.save(bytes_io, array)
    bytes_io.seek(0)
    blob_name = f"{job_id}/{name}.npy" if job_id else f"{name}.npy"
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    blob_client.upload_blob(bytes_io.getvalue(), overwrite=True)

# Save data to blob storage
save_array_to_blob(x_train, "x_train")
save_array_to_blob(y_train, "y_train")
save_array_to_blob(x_test, "x_test")
save_array_to_blob(y_test, "y_test")

print(f"MNIST data uploaded to Azure Blob storage successfully. Container: {container_name}, Job ID: {job_id if job_id else 'Not specified'}")