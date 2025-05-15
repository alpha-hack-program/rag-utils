import os
import json
import logging
import time
import argparse
import hashlib

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

from shared.rag_utils import calculate_md5, is_processed, mark_file_processed

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

        # Log the file converted and the output file path
        logging.info(
            f"Document {conv_result.input.file} converted successfully. "
            f"Output file: {output_dir / f'{doc_filename}.json'}"
        )

        if conv_result.document:
            conv_result.document.save_as_json(
                output_dir / f"{doc_filename}.json",
                image_mode=ImageRefMode.PLACEHOLDER,
            )
            conv_result.document.save_as_doctags(
                output_dir / f"{doc_filename}.doctags.txt"
            )
            conv_result.document.save_as_markdown(
                output_dir / f"{doc_filename}.md",
                image_mode=ImageRefMode.PLACEHOLDER,
            )
        else:
            _log.error(
                f"Document {conv_result.input.file} was converted but no document was created."
            )
            with (output_dir / f"{doc_filename}.json").open("w") as fp:
                json.dump(conv_result.to_dict(), fp, indent=2)
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

def main():
    # Validate and set level
    if LOG_LEVEL_STR in VALID_LOG_LEVELS:
        log_level = getattr(logging, LOG_LEVEL_STR)
    else:
        print(f"Invalid LOG_LEVEL '{LOG_LEVEL_STR}' - defaulting to WARNING")
        log_level = logging.WARNING

    logging.basicConfig(level=log_level)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Parse outputdir, basedir, and a list of files.")
    
    parser.add_argument(
        '--outputdir',
        required=True,
        help='Path to the output directory'
    )
    
    parser.add_argument(
        '--inputdir',
        required=True,
        help='Path to the input directory'
    )
    
    args = parser.parse_args()

    print(f"Input directory: {args.inputdir}")
    print(f"Output directory: {args.outputdir}")

    # Start the timer
    start_time = time.time()

    # Convert the documents
    success, partial_success, failure = _docling_convert(
        input_dir=Path(args.inputdir),
        output_dir=Path(args.outputdir),
    )

    # Log the conversion results
    _log.info(
        f"Successfully converted: {success}"
    )
    _log.info(
        f"Partially converted: {partial_success}"
    )
    _log.info(
        f"Failed to convert: {failure}"
    )

    # Stop the timer
    end_time = time.time() - start_time

    _log.info(f"Document conversion complete in {end_time:.2f} seconds.")
    
if __name__ == "__main__":
    main()