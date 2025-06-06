# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

import os

from kfp import dsl

from src.docling_chunker import docling_chunker

COMPONENT_NAME=os.getenv("COMPONENT_NAME")
print(f"COMPONENT_NAME: {COMPONENT_NAME}")

# This pipeline will download training dataset, download the model, test the model and if it performs well, 
# upload the model to another S3 bucket.
@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def pipeline(
    root_mount_path: str,
    input_dir_name: str,
    output_dir_name: str,
    tokenizer_embed_model_id: str  = '',
    tokenizer_max_tokens: int = 200,
    merge_peers: bool = True,
    ):

    root_mount_path = "/opt/app/src"

    # Train the model
    docling_chunker_task = docling_chunker(
        root_mount_path=root_mount_path,
        input_dir_name=input_dir_name,
        output_dir_name=output_dir_name,
        tokenizer_embed_model_id=tokenizer_embed_model_id,
        tokenizer_max_tokens=tokenizer_max_tokens,
        merge_peers=merge_peers
    ).set_caching_options(False) # type: ignore

if __name__ == '__main__':
    from shared.kubeflow import compile_and_upsert_pipeline
    
    import os

    pipeline_package_path = __file__.replace('.py', '.yaml')

    # Pipeline name
    pipeline_name=f"{COMPONENT_NAME}_pl"

    compile_and_upsert_pipeline(
        pipeline_func=pipeline, # type: ignore
        pipeline_package_path=pipeline_package_path,
        pipeline_name=pipeline_name
    )