# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

import os

from kfp import dsl

from src.setup_storage import setup_storage

COMPONENT_NAME=os.getenv("COMPONENT_NAME")
print(f"COMPONENT_NAME: {COMPONENT_NAME}")

# This pipeline will download training dataset, download the model, test the model and if it performs well, 
# upload the model to another S3 bucket.
@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def pipeline(
    pvc_name: str = "my-pvc",
    size_in_gi: int = 1
    ):

    # Sync files from S3 to local
    setup_storage_task = setup_storage(
        pvc_name=pvc_name,
        size_in_gi=size_in_gi
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