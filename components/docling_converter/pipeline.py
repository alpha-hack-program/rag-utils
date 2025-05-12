# DOCS: https://www.kubeflow.org/docs/components/pipelines/user-guides/components/ 

import os

from kfp import dsl

from src.docling_converter import docling_chunker

COMPONENT_NAME=os.getenv("COMPONENT_NAME")
print(f"COMPONENT_NAME: {COMPONENT_NAME}")

# This pipeline will download training dataset, download the model, test the model and if it performs well, 
# upload the model to another S3 bucket.
@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def pipeline(
    input_doc_paths: str = "/usr/local/src/kfp/components/tests/data/2203.01017v2.pdf,/usr/local/src/kfp/components/tests/data/GenAI_POC_Process_Guidebook.docx",
    output_dir: str = "/tmp/docling-converter"    
):

    # Convert documents to JSON(docling format) and markdown
    train_model_task = docling_chunker(
       input_doc_paths=input_doc_paths,
        output_dir=output_dir, 
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