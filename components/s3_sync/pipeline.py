# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

import os

from kfp import dsl

from src.s3_sync import s3_sync

COMPONENT_NAME=os.getenv("COMPONENT_NAME")
print(f"COMPONENT_NAME: {COMPONENT_NAME}")

# This pipeline will download training dataset, download the model, test the model and if it performs well, 
# upload the model to another S3 bucket.
@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def pipeline(
    folder: str = "collections",
    root_mount_path: str = "/tmp",
    local_folder: str = "collections",
    force: bool = False):

    # Sync files from S3 to local
    s3_sync_task = s3_sync(
        folder=folder,
        root_mount_path=root_mount_path,
        local_folder=local_folder,
        force=force
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