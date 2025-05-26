# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

# Pipeline to load documents from an S3 bucket into Milvus using pymilvus => mv

import os

from kfp import dsl

from kfp import kubernetes
from kfp.components import load_component_from_file

from components.setup_storage import setup_storage as setup_storage_component
from components.s3_sync import s3_sync as s3_sync_component
from components.docling_converter import docling_converter as docling_converter_component
from components.docling_chunker import docling_chunker as docling_chunker_component
from components.add_chunks_to_milvus import add_chunks_to_milvus as add_chunks_to_milvus_component

# # Load the register model from a url
# REGISTER_MODEL_COMPONENT_URL = "https://raw.githubusercontent.com/alpha-hack-program/model-serving-utils/refs/heads/main/components/register_model/src/component_metadata/register_model.yaml"
# register_model_component = kfp.components.load_component_from_url(REGISTER_MODEL_COMPONENT_URL)

# Load train_yolo component
# SETUP_STORAGE_COMPONENT_FILE_PATH='components/setup_storage/src/component_metadata/setup_storage.yaml'
# setup_storage_component = load_component_from_file(SETUP_STORAGE_COMPONENT_FILE_PATH)

# This pipeline will download evaluation data, download the model, test the model and if it performs well, 
# upload the model to the runtime S3 bucket and refresh the runtime deployment.
@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def pipeline(
    root_mount_path: str = "/opt/app-root/src",
    documents_bucket_folder: str = "documents",
    milvus_collection_name: str = "chunks",
    images_dataset_pvc_name: str = "images-dataset-pvc",
    images_dataset_pvc_size_in_gi: int = 10,
    documents_dir_name: str = "documents",
    converted_dir_name: str = "converted",
    chunks_dir_name: str = "chunks",
    force: bool = False, # Force download even if the file exists locally
):
    
    # Define the datasets volume
    datasets_pvc_name = images_dataset_pvc_name
    datasets_pvc_size_in_gi = images_dataset_pvc_size_in_gi

    # Define the root mount path
    setup_storage_task = setup_storage_component(
        pvc_name=datasets_pvc_name, 
        size_in_gi=datasets_pvc_size_in_gi
    ).set_display_name("setup_storage")

    # Sync files from S3 to local
    s3_sync_task = s3_sync_component(
        folder=documents_bucket_folder,
        root_mount_path=root_mount_path,
        local_folder=documents_dir_name,
        force=force
    ).after(setup_storage_task).set_display_name("s3_sync").set_caching_options(False)

    # Set document input and output directories
    documents_input_dir = root_mount_path / documents_dir_name
    converted_output_dir = root_mount_path / converted_dir_name

    # Convert documents using docling
    docling_converter_task = docling_converter_component(
        input_dir=documents_input_dir,
        output_dir=converted_output_dir,
    ).after(s3_sync_task).set_display_name("docling_converter").set_caching_options(False)

    # Set document input and output directories
    converted_input_dir = root_mount_path / converted_dir_name
    chunks_output_dir = root_mount_path / chunks_dir_name

    # Chunk documents using docling
    docling_chunker_task = docling_chunker_component(
        input_dir=converted_input_dir,
        output_dir=chunks_output_dir,
    ).after(docling_converter_task).set_display_name("docling_chunker").set_caching_options(False)

    # Add chunks to vector store
    add_chunks_to_vector_store_task = add_chunks_to_milvus_component(
        input_dir=chunks_output_dir,
        milvus_collection_name=milvus_collection_name,
    ).after(docling_chunker_task).set_display_name("insert_chunks").set_caching_options(False)
        

    # Set the kubernetes secret to be used in the get_chunks_from_documents task
    kubernetes.use_secret_as_env(
        task=s3_sync_task,
        secret_name='aws-connection-documents',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
        })

    # Set the kubernetes secret to be used in the add_chunks_to_milvus task
    kubernetes.use_secret_as_env(
        task=add_chunks_to_vector_store_task,
        secret_name='milvus-connection-documents',
        secret_key_to_env={
            'MILVUS_DATABASE': 'MILVUS_DATABASE',
            'MILVUS_HOST': 'MILVUS_HOST',
            'MILVUS_PORT': 'MILVUS_PORT',
            'MILVUS_USERNAME': 'MILVUS_USERNAME',
            'MILVUS_PASSWORD': 'MILVUS_PASSWORD',
        })
    
    # Mount the PVC to task s3_sync_task
    kubernetes.mount_pvc(
        s3_sync_task,
        pvc_name=datasets_pvc_name,
        mount_path=root_mount_path
    )

    # Mount the PVC to docling_converter_task
    kubernetes.mount_pvc(
        docling_converter_task,
        pvc_name=datasets_pvc_name,
        mount_path=root_mount_path
    )
    # Mount the PVC to docling_chunker_task
    kubernetes.mount_pvc(
        docling_chunker_task,
        pvc_name=datasets_pvc_name,
        mount_path=root_mount_path
    )
    # Mount the PVC to add_chunks_to_vector_store_task
    kubernetes.mount_pvc(
        add_chunks_to_vector_store_task,
        pvc_name=datasets_pvc_name,
        mount_path=root_mount_path
    )
    
if __name__ == '__main__':
    from shared.kubeflow import compile_and_upsert_pipeline
    
    import os

    pipeline_package_path = __file__.replace('.py', '.yaml')

    # Pipeline name
    pipeline_name=os.path.basename(__file__).replace('.py', '')

    compile_and_upsert_pipeline(
        pipeline_func=pipeline,
        pipeline_package_path=pipeline_package_path,
        pipeline_name=pipeline_name
    )