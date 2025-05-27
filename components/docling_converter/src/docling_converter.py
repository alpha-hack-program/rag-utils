import os
import json
import logging

from pathlib import Path

from tqdm import tqdm

from docling_core.types.doc import ImageRefMode

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, FormatOption, PdfFormatOption, HTMLFormatOption, MarkdownFormatOption
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PipelineOptions, PdfPipelineOptions
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.backend.asciidoc_backend import AsciiDocBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.backend.html_backend import HTMLDocumentBackend
from docling.backend.json.docling_json_backend import DoclingJSONBackend
from docling.backend.md_backend import MarkdownDocumentBackend
from docling.backend.msexcel_backend import MsExcelDocumentBackend
from docling.backend.mspowerpoint_backend import MsPowerpointDocumentBackend
from docling.backend.msword_backend import MsWordDocumentBackend
from docling.backend.xml.uspto_backend import PatentUsptoDocumentBackend

from shared.rag_utils import is_processed, mark_file_processed

from kfp import dsl

NAMESPACE = os.environ.get("NAMESPACE", "default")
COMPONENT_NAME=os.getenv("COMPONENT_NAME", f"docling_chunker")
BASE_IMAGE=os.getenv("BASE_IMAGE", "quay.io/modh/runtime-images:runtime-pytorch-ubi9-python-3.9-20241111")
REGISTRY=os.environ.get("REGISTRY", f"image-registry.openshift-image-registry.svc:5000/{NAMESPACE}")
TAG=os.environ.get("TAG", f"latest")
TARGET_IMAGE=f"{REGISTRY}/{COMPONENT_NAME}:{TAG}"

# Pip package versions
LOAD_DOTENV_PIP_VERSION="0.1.0"
DOCLING_PIP_VERSION="2.31.0"

# MAX_INPUT_DOCS is the value of MAX_INPUT_DOCS environment variable or 20
MAX_INPUT_DOCS = int(os.environ.get("MAX_INPUT_DOCS", 2))

# Acceptable input formats with extensions and their corresponding FormatOption
# for example: InputFormat.PDF (["pdf", "PDF"]) -> PdfFormatOption
ALLOWED_INPUT_FORMATS = {
    InputFormat.PDF: [ "pdf", "PDF"],
    InputFormat.DOCX: ["docx", "DOCX"],
    InputFormat.PPTX: ["pptx", "PPTX"],
    InputFormat.MD: ["md","MD"],
    InputFormat.HTML: ["html","HTML"],
}

# Allowed log levels
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

# Read from environment variable
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "WARNING").upper()

# Set the log level for the 'docling.utils' logger to ERROR
logging.getLogger('docling').setLevel(logging.ERROR)

# Create a logger for this module
_log = logging.getLogger(__name__)

def export_document(
    conv_result: ConversionResult,
    input_dir: Path,
    output_dir: Path,
) -> bool:
    """
    Export the converted documents to the specified output directory.
    Args:
        conv_result: ConversionResult: The conversion result object containing the conversion status and document.
        output_dir (Path): Directory to save the converted documents.
    Returns:
        Tuple[int, int, int]: Counts of successful, partially successful, and failed conversions.
    """

    # Log entry
    _log.debug(f">>> Exporting documents to {output_dir}")

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check the conversion status and handle accordingly
    if conv_result.status == ConversionStatus.SUCCESS:
        doc_filename = conv_result.input.file.stem

        # Log the document filename
        _log.debug(f"Document filename: {doc_filename}")

        # Extract the relative path of the directory the document file is in compared to the input directory
        doc_relative_path = conv_result.input.file.relative_to(input_dir).parent

        # Log the relative path
        _log.debug(f"Document relative path: {doc_relative_path}")

        # Generate the output directory path
        doc_path = output_dir / doc_relative_path

        # Log the output directory
        _log.debug(f"Output directory: {doc_path}")

        # Create the output directory if it doesn't exist
        doc_path.mkdir(parents=True, exist_ok=True)
        
        # Log the file converted and the output file path
        logging.info(
            f"Document {conv_result.input.file} converted successfully. "
            f"Writing Docling document to: {doc_path / f'{doc_filename}.json'}"
        )

        if conv_result.document:
            conv_result.document.save_as_json(
                doc_path / f"{doc_filename}.json",
                image_mode=ImageRefMode.PLACEHOLDER,
            )
            conv_result.document.save_as_doctags(
                doc_path / f"{doc_filename}.doctags.txt"
            )
            conv_result.document.save_as_markdown(
                doc_path / f"{doc_filename}.md",
                image_mode=ImageRefMode.PLACEHOLDER,
            )
        else:
            _log.error(
                f"Document {conv_result.input.file} was converted but no document was created."
            )
            # with (doc_path / f"{doc_filename}.json").open("w") as fp:
            #     json.dump(conv_result.to_dict(), fp, indent=2)
            return False

    return True

def get_allowed_list_of_input_formats() -> list[InputFormat]:
    """
    Get the allowed input formats.
    Returns:
        list[InputFormat]: List of allowed input formats.
    """
    return list(ALLOWED_INPUT_FORMATS.keys())

def generate_format_options(
    extensions: set[str],   
    pipeline_options: PipelineOptions     
) -> dict[InputFormat, FormatOption]:
    """
    Generate format options based on the provided extensions.
    Args:
        extensions (set[str]): Set of file extensions.
    Returns:
        dict[InputFormat, FormatOption]: Dictionary mapping InputFormat to FormatOption.
    """
    
    format_options = {}
    for ext in extensions:
        for fmt, _ext in ALLOWED_INPUT_FORMATS.items():
            if ext in _ext:
                if ext in ALLOWED_INPUT_FORMATS[InputFormat.PDF]:
                    format_options[fmt] = PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                elif ext in ALLOWED_INPUT_FORMATS[InputFormat.DOCX]:
                    format_options[fmt] = FormatOption()
                elif ext in ALLOWED_INPUT_FORMATS[InputFormat.PPTX]:
                    format_options[fmt] = FormatOption(
                        pipeline_options=pipeline_options
                    )
                elif ext in ALLOWED_INPUT_FORMATS[InputFormat.MD]:
                    format_options[fmt] = MarkdownFormatOption(
                        pipeline_options=pipeline_options
                    )
                elif ext in ALLOWED_INPUT_FORMATS[InputFormat.HTML]:
                    format_options[fmt] = HTMLFormatOption(
                        pipeline_options=pipeline_options
                    )
                else:
                    raise ValueError(
                        f"Unsupported file extension: {ext}. Supported extensions are: {', '.join(ALLOWED_INPUT_FORMATS[fmt])}"
                    )

    return format_options

# Function to convert documents using Docling
# This function takes a list of input document paths and an output directory
# and converts the documents using the specified pipeline options.
def _docling_convert(
    input_dir: Path,
    output_dir: Path
) -> tuple[list[str], list[str], list[str]]:
    """
    Convert documents using Docling.
    Args:
        input_dir (Path): Directory containing the input documents.
        output_dir (Path): Directory to save the converted documents.
    Returns:
        Tuple[list[str], list[str], list[str]]: Lists of successfully converted, partially converted, and failed documents.
    """

    # Log input directory
    _log.info(f"Input directory: {input_dir}")
    # Log output directory
    _log.info(f"Output directory: {output_dir}")
    
    # Check if the input directory exists
    if not input_dir.exists():
        _log.error(f"Input directory {input_dir} does not exist.")
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

    # Check if all existing paths have supported extensions and separate them into valid and invalid extensions
    valid_extensions = [
        ext for fmt, exts in ALLOWED_INPUT_FORMATS.items() for ext in exts
    ]

    # Log the valid extensions
    _log.info(f"Valid extensions: {valid_extensions}")

    # Iterate over all files in the input directory and its subdirectories and print the file paths and suffixes
    input_doc_paths = []
    for file in input_dir.rglob("*"):
        # Check if the file is a file and not a directory and if it has a suffix
        if file.is_file() and file.suffix:
            _log.debug(f"File: {file}, Suffix: {file.suffix}")
            # Check if any parent directory ends with ".chunks"
            if any(part.endswith('.chunks') for part in file.parts):
                _log.debug(f"File {file} is in a .chunks directory, skipping.")
                continue
            # Check if the file has a valid extension
            if file.suffix[1:] in valid_extensions:
                _log.debug(f"File {file} has a valid extension: {file.suffix}")
                if not is_processed(file):
                    _log.debug(f"File {file} has not been processed, adding to input_doc_paths.")
                    # Add the file to the list of input document paths
                    input_doc_paths.append(file)
            else:
                _log.debug(f"File {file} has an invalid extension: {file.suffix}")

    # Log the number of input documents
    _log.info(f"Found {len(input_doc_paths)} input documents in the directory.")
    # Log the input document paths
    _log.debug(f"Input document paths: {', '.join(map(str, input_doc_paths))}")

    # Create batches of MAX_INPUT_DOCS documents out of the input existing_paths
    batches = [
        input_doc_paths[i : i + MAX_INPUT_DOCS] for i in range(0, len(input_doc_paths), MAX_INPUT_DOCS)
    ]

    # Log the number of batches created
    _log.info(  
        f"Created {len(batches)} batches of documents, each with a maximum of {MAX_INPUT_DOCS} documents."
    )

    # Log batch content if log level is DEBUG
    if _log.isEnabledFor(logging.DEBUG):
        for i, batch in enumerate(batches):
            _log.debug(
                f"Batch {i + 1}: {', '.join(map(str, batch))}"
            )

    # Create PipelineOptions
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.AUTO
    )
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    # Create the format options for the converter
    format_options = {
        InputFormat.XLSX: FormatOption(
            pipeline_cls=SimplePipeline, backend=MsExcelDocumentBackend
        ),
        InputFormat.DOCX: FormatOption(
            pipeline_cls=SimplePipeline, backend=MsWordDocumentBackend
        ),
        InputFormat.PPTX: FormatOption(
            pipeline_cls=SimplePipeline, backend=MsPowerpointDocumentBackend
        ),
        InputFormat.MD: FormatOption(
            pipeline_cls=SimplePipeline, backend=MarkdownDocumentBackend
        ),
        InputFormat.ASCIIDOC: FormatOption(
            pipeline_cls=SimplePipeline, backend=AsciiDocBackend
        ),
        InputFormat.HTML: FormatOption(
            pipeline_cls=SimplePipeline, backend=HTMLDocumentBackend
        ),
        InputFormat.XML_USPTO: FormatOption(
            pipeline_cls=SimplePipeline, backend=PatentUsptoDocumentBackend
        ),
        InputFormat.IMAGE: FormatOption(
            pipeline_cls=StandardPdfPipeline, backend=DoclingParseV2DocumentBackend
        ),
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        ),
        InputFormat.JSON_DOCLING: FormatOption(
            pipeline_cls=SimplePipeline, backend=DoclingJSONBackend
        ),
    }

    # Create the DocumentConverter with the format options
    converter = DocumentConverter(
        allowed_formats=get_allowed_list_of_input_formats(),
        format_options=format_options
    )

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # List of successfully converted documents
    succesfully_converted_documents = []
    # List of partially converted documents
    partially_converted_documents = []
    # List of failed documents
    failed_documents = []

    # Convert the documents in batches, use tqdm to show progress
    for batch in tqdm(batches):
        # Convert the documents in the batch
        conv_results = converter.convert_all(
            batch,
            raises_on_error=False,  # to let conversion run through all and examine results at the end
        )

        # Iterate over the conversion results and categorize them
        for conv_res in conv_results:
            # Check if the conversion was successful
            if conv_res.status == ConversionStatus.SUCCESS:
                export_document(
                    conv_res,
                    input_dir=input_dir,
                    output_dir=output_dir,
                )
                succesfully_converted_documents.append(conv_res.input.file)
                # Mark the file as processed by creating a file with the same name + "{hash}" in the same directory
                mark_file_processed(conv_res.input.file)
            # Check the conversion status and categorize accordingly
            elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
                partially_converted_documents.append(conv_res.input.file)
            else:
                failed_documents.append(conv_res.input.file)

    # Log the total number of documents processed
    _log.info(
        f"Processed a total of {len(input_doc_paths)} documents"
    )
    # Return the counts of successful, partial success, and failure
    success_count = len(succesfully_converted_documents)
    partial_success_count = len(partially_converted_documents)
    failure_count = len(failed_documents)
    # Log the total number of documents processed
    _log.info(
        f"Processed {success_count + partial_success_count + failure_count} docs, "
        f"of which {failure_count} failed "
        f"and {partial_success_count} were partially converted."
    )
    # Return the lists of successful, partial success, and failure conversions
    return (
        succesfully_converted_documents,
        partially_converted_documents,
        failed_documents,
    )

# Function that 
@dsl.component(
    base_image=BASE_IMAGE,
    target_image=TARGET_IMAGE,
    packages_to_install=[f"docling[vlm]=={DOCLING_PIP_VERSION}", f"load_dotenv=={LOAD_DOTENV_PIP_VERSION}"]
)
def docling_converter(
    root_mount_path: str,
    input_dir_name: str,
    output_dir_name: str
) -> str:
    """
    Convert documents using Docling.
    Args:
        root_mount_path (str): The root mount path where the input and output directories are located.
        input_dir_name (str): The name of the input directory containing documents to convert.
        output_dir_name (str): The name of the output directory where converted documents will be saved.
    Returns:
        list[str]: Lists of successfully converted
    """
    
    # Convert input_dir_name and output_dir_name to Path objects
    input_dir = Path(root_mount_path) / input_dir_name
    output_dir = Path(root_mount_path) / output_dir_name
    # Log the input and output directories
    _log.info(f"Input directory: {input_dir}")
    _log.info(f"Output directory: {output_dir}")

    # Convert input_dir to a Path object and fail if it can't be converted
    if not isinstance(input_dir, Path):
        raise ValueError(f"Input directory {input_dir} is not a valid path.")

    # Check if the input directory exists
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist.")

    # Convert output_dir to a Path object and fail if it can't be converted
    if not isinstance(output_dir, Path):
        raise ValueError(f"Output directory {output_dir} is not a valid path.")

    # Convert the documents
    succesfully_converted_documents, partially_converted_documents, failed_documents = _docling_convert(
        input_dir=input_dir,
        output_dir=output_dir
    )
    
    # return the lists as a json string
    return json.dumps({
        "success": [str(path) for path in succesfully_converted_documents],
        "partial_success": [str(path) for path in partially_converted_documents],
        "failure": [str(path) for path in failed_documents]
    })
    
if __name__ == "__main__":
    # Generate and save the component YAML file
    component_package_path = __file__.replace('.py', '.yaml')

    docling_converter.save_component_yaml(component_package_path)
