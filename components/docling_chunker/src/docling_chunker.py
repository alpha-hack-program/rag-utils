import os
import shutil

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption


from shared.kubeflow import get_token

from kfp import compiler

from kfp.dsl import Output, Metrics, OutputPath
from kfp import dsl

NAMESPACE = os.environ.get("NAMESPACE", "default")
COMPONENT_NAME=os.getenv("COMPONENT_NAME", f"docling_chunker")
BASE_IMAGE=os.getenv("BASE_IMAGE", "quay.io/modh/runtime-images:runtime-pytorch-ubi9-python-3.9-20241111")
REGISTRY=os.environ.get("REGISTRY", f"image-registry.openshift-image-registry.svc:5000/{NAMESPACE}")
TAG=os.environ.get("TAG", f"latest")
TARGET_IMAGE=f"{REGISTRY}/{COMPONENT_NAME}:{TAG}"

LOAD_DOTENV_PIP_VERSION="0.1.0"

# Function to convert a PDF file to Markdown text
def convert_pdf_to_markdown(file_path: str) -> str:
    """
    Converts a PDF file at `file_path` to Markdown text and returns it.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    # Enable the profiling to measure the time spent
    settings.debug.profile_pipeline_timings = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Convert the document
    conversion_result = converter.convert(file_path)
    document = conversion_result.document

    # Print total conversion time
    conversion_secs = conversion_result.timings["pipeline_total"].times
    print(f"Conversion secs: {conversion_secs}")

    # Return the resulting Markdown text
    return document.export_to_markdown()

# Function that 
@dsl.component(
    base_image=BASE_IMAGE,
    target_image=TARGET_IMAGE,
    packages_to_install=[f"ultralytics=={ULTRALYTICS_PIP_VERSION}", f"load_dotenv=={LOAD_DOTENV_PIP_VERSION}", 
                         f"numpy=={NUMPY_PIP_VERSION}", f"mlflow=={MLFLOW_PIP_VERSION}", 
                         f"onnxruntime=={ONNXRUNTIME_PIP_VERSION}", f"onnxslim=={ONNXSLIM_PIP_VERSION}",
                         f"botocore=={BOTOCORE_PIP_VERSION}",
                         f"boto3=={BOTO3_PIP_VERSION}"]
)
def docling_chunker(
    model_name: str, 
    image_size: int, 
    epochs: int, 
    batch_size: int,
    optimizer: str,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    confidence_threshold: float,
    iou_threshold: float,
    label_smoothing: float,
    experiment_name: str,
    run_name: str,
    tracking_uri: str,
    root_mount_path: str,
    images_dataset_name: str,
    images_datasets_root_folder: str,
    images_dataset_yaml: str,
    models_root_folder: str,
    metric_value_output: OutputPath(float), # type: ignore
    model_name_output: OutputPath(str), # type: ignore
    results_output_metrics: Output[Metrics]
):      
    datasets_endpoint_url = os.environ.get('DATASETS_AWS_S3_ENDPOINT')
    datasets_region_name = os.environ.get('DATASETS_AWS_DEFAULT_REGION')
    datasets_bucket_name = os.environ.get('DATASETS_AWS_S3_BUCKET')
    datasets_aws_access_key_id = os.environ.get('DATASETS_AWS_ACCESS_KEY_ID')
    datasets_aws_secret_access_key = os.environ.get('DATASETS_AWS_SECRET_ACCESS_KEY')
    images_dataset_s3_key = os.environ.get('IMAGES_DATASET_S3_KEY')

    print(f"DATASETS S3: endpoint_url {datasets_endpoint_url}")
    print(f"DATASETS S3: bucket_name {datasets_bucket_name}")
    print(f"DATASETS S3: images_dataset_s3_key {images_dataset_s3_key}")

    models_endpoint_url = os.environ.get('MODELS_AWS_S3_ENDPOINT')
    models_region_name = os.environ.get('MODELS_AWS_DEFAULT_REGION')
    models_bucket_name = os.environ.get('MODELS_AWS_S3_BUCKET')
    models_aws_access_key_id = os.environ.get('MODELS_AWS_ACCESS_KEY_ID')
    models_aws_secret_access_key = os.environ.get('MODELS_AWS_SECRET_ACCESS_KEY')
    ultralitics_base_models_folder = os.environ.get('ULTRALYTICS_BASE_MODELS_FOLDER')

    print(f"MODELS S3: endpoint_url {datasets_endpoint_url}")
    print(f"MODELS S3: bucket_name {datasets_bucket_name}")
    print(f"MODELS S3: images_dataset_s3_key {images_dataset_s3_key}")

    print(f"tracking_uri {tracking_uri}")
    print(f"experiment_name {experiment_name}")
    print(f"images_dataset_name {images_dataset_name}")
    print(f"images_datasets_root_folder {images_datasets_root_folder}")
    print(f"images_dataset_yaml {images_dataset_yaml}")
    print(f"models_root_folder {models_root_folder}")
    print(f"root_mount_path {root_mount_path}")

    # If root_mount_path is not set or doesn't exist, raise a ValueError
    if not root_mount_path or not os.path.exists(root_mount_path):
        raise ValueError(f"Root mount path '{root_mount_path}' does not exist")

    # Set the images dataset folder
    images_dataset_folder = os.path.join(root_mount_path, images_datasets_root_folder, images_dataset_name)

    # If the images dataset folder doesn't exist, raise a ValueError
    if not os.path.exists(images_dataset_folder):
        raise ValueError(f"Images dataset folder {images_dataset_folder} does not exist")

    # Set the models folder
    models_folder = os.path.join(root_mount_path, models_root_folder)

    # Make sure the models folder exists
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    # Set the images dataset YAML path
    images_dataset_yaml_path = os.path.join(images_dataset_folder, images_dataset_yaml)
    print(f"Checking if {images_dataset_yaml_path} exists")
    if not os.path.exists(images_dataset_yaml_path):
        raise ValueError(f"Dataset YAML file {images_dataset_yaml} not found in {images_dataset_folder}")
    print(f"Dataset YAML file found in {images_dataset_yaml_path}")

    # Set the MLflow tracking URI and experiment
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name

    # Get the kubernetes token in a string
    os.environ["MLFLOW_TRACKING_TOKEN"] = get_token()

    # Update a setting
    settings.update({"mlflow": True})

    # Set device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using device: MPS")
    print(f"Using device: {device}")

    # Reset settings to default values
    settings.reset()

    # Download the base yolo model
    yolo_model_path = download_yolo_model(endpoint_url=models_endpoint_url, 
                                          region_name=models_region_name, 
                                          bucket_name=models_bucket_name,
                                          base_models_folder=ultralitics_base_models_folder,
                                          model_file_name=f"{model_name}.pt",
                                          aws_access_key_id=models_aws_access_key_id, 
                                          aws_secret_access_key=models_aws_secret_access_key)

    # Check if yolo_model_path exists
    
    # Load the model
    model = YOLO(f'{yolo_model_path}')

    # Set the run name
    train_run_name = f"{run_name}-train"
    print(f"Current run name: {train_run_name}")

    # Dataset yaml path
    images_dataset_folder = os.path.join(root_mount_path, images_datasets_root_folder, images_dataset_name)
    images_dataset_yaml_path = os.path.join(images_dataset_folder, images_dataset_yaml)

    print(f"Checking if {images_dataset_yaml_path} exists")
    if not os.path.exists(images_dataset_yaml_path):
        raise ValueError(f"Dataset YAML file {images_dataset_yaml} not found in {images_dataset_folder}")
    print(f"Dataset YAML file found in {images_dataset_yaml_path}")

    # Start the MLflow run for training
    with mlflow.start_run(run_name=train_run_name) as training_mlrun:
        mlflow.log_param("dataset_file", f"{datasets_endpoint_url}/{datasets_bucket_name}/{images_dataset_s3_key}")
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("dataset_name", images_dataset_name)
        mlflow.log_param("datasets_root_folder", images_datasets_root_folder)
        mlflow.log_param("dataset_yaml", images_dataset_yaml)
        mlflow.log_param("device", device)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("imgsz", image_size)

        # Create a temporary directory
        yolo_temp_dir = tempfile.mkdtemp()

        print(f"Training model {model_name} with dataset {images_dataset_yaml_path}.")
        results = model.train(
                data=images_dataset_yaml_path,
                epochs=epochs,
                imgsz=image_size,
                batch=batch_size,
                optimizer=optimizer,
                lr0=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                conf=confidence_threshold,
                iou=iou_threshold,
                label_smoothing=label_smoothing,
                device=device,
                project=yolo_temp_dir
            )
        
        metric_value = 0.0
        if hasattr(results, 'box'):
            results_output_metrics.log_metric("training/map", results.box.map if results.box.map is not None else 0.0)
            results_output_metrics.log_metric("training/map50", results.box.map50 if results.box.map50 is not None else 0.0)
            results_output_metrics.log_metric("training/map75", results.box.map75 if results.box.map75 is not None else 0.0)
            results_output_metrics.log_metric("training/mp", results.box.mp if results.box.mp is not None else 0.0)
            results_output_metrics.log_metric("training/mr", results.box.mr if results.box.mr is not None else 0.0)
            results_output_metrics.log_metric("training/nc", results.box.nc if results.box.nc is not None else 0.0)

            metric_value = results.box.map
        else:
            print("No box attribute in the results!!!")

        # Save the trained model
        print(f"Saving model to {models_folder}")
        trained_model_name = f"{train_run_name}"
        print(f"Trained model name: {trained_model_name}")
        trained_model_pt_path = os.path.join(models_folder, f"{trained_model_name}.pt")
        print(f"Saving model to {trained_model_pt_path}")
        model.save(trained_model_pt_path)
        print(f"Model saved to {trained_model_pt_path}")

        #  If the trained model was not saved, raise a ValueError
        if not os.path.exists(trained_model_pt_path):
            raise ValueError(f"Model was not saved at {trained_model_pt_path}")
        
        # End the run
        mlflow.end_run()

        # Start the MLflow run for validation
        val_run_name = f"{run_name}-val"
        print(f"Current run name: {val_run_name}")
        with mlflow.start_run(run_name=val_run_name):
            # Validate the model    
            validation_results = model.val()

            # If the results have the box attribute, log the metrics
            if hasattr(validation_results, 'box'):
                mlflow.log_metric("val/map", validation_results.box.map)
                mlflow.log_metric("val/map50", validation_results.box.map50)
                mlflow.log_metric("val/map75", validation_results.box.map75)
                mlflow.log_metric("val/mp", validation_results.box.mp)
                mlflow.log_metric("val/mr", validation_results.box.mr)
                mlflow.log_metric("val/nc", validation_results.box.nc)
                results_output_metrics.log_metric("val/map", validation_results.box.map)
                results_output_metrics.log_metric("val/map50", validation_results.box.map50)
                results_output_metrics.log_metric("val/map75", validation_results.box.map75)
                results_output_metrics.log_metric("val/mp", validation_results.box.mp)
                results_output_metrics.log_metric("val/mr", validation_results.box.mr)
                results_output_metrics.log_metric("val/nc", validation_results.box.nc)
            else:
                print("No box attribute in the results!!!")
            
        # Convert the model to ONNX
        trained_model_onnx_path_tmp = model.export(format="onnx")
        if not trained_model_onnx_path_tmp:
            print("Failed to export model to ONNX format")

        # Save the onnx model
        trained_model_onnx_path = os.path.join(models_folder, f"{trained_model_name}.onnx")
        print(f"Copying {trained_model_onnx_path_tmp} to {trained_model_onnx_path}")
        shutil.copy(trained_model_onnx_path_tmp, trained_model_onnx_path)
        print(f"Copied {trained_model_onnx_path_tmp} to {trained_model_onnx_path}")

        # Output the model_name
        with open(model_name_output, 'w') as f:
            f.write(trained_model_name)
        
        # Output the metric_value
        with open(metric_value_output, 'w') as f:
            f.write(str(metric_value))

    if not training_mlrun:
        raise ValueError("MLflow run was not started")

if __name__ == "__main__":
    # Generate and save the component YAML file
    component_package_path = __file__.replace('.py', '.yaml')

    compiler.Compiler().compile(
        pipeline_func=docling_chunker,
        package_path=component_package_path
    )
