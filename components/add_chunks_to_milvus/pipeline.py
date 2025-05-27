# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

import os

from kfp import dsl

from src.add_chunks_to_milvus import add_chunks_to_milvus

COMPONENT_NAME=os.getenv("COMPONENT_NAME")
print(f"COMPONENT_NAME: {COMPONENT_NAME}")

# This pipeline will download training dataset, download the model, test the model and if it performs well, 
# upload the model to another S3 bucket.
@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def pipeline(
    milvus_collection_name: str = "chunks",
    root_mount_path: str = "/tmp",
    input_dir_name: str = "collections",
    force: bool = False):

    # Add chunks to Milvus
    add_chunks_to_milvus_task = add_chunks_to_milvus(
        root_mount_path=root_mount_path,
        input_dir_name=input_dir_name,
        milvus_collection_name=milvus_collection_name,
    ).set_caching_options(False)

if __name__ == '__main__':
    from shared.kubeflow import compile_and_upsert_pipeline
    
    import os

    pipeline_package_path = __file__.replace('.py', '.yaml')

    # Pipeline name
    pipeline_name=f"{COMPONENT_NAME}_pl"

    compile_and_upsert_pipeline(
        pipeline_func=pipeline,
        pipeline_package_path=pipeline_package_path,
        pipeline_name=pipeline_name
    )