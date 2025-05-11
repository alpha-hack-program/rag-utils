import os
import json
import logging
import time

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
from docling.backend.xml.pubmed_backend import PubMedDocumentBackend
from docling.backend.xml.uspto_backend import PatentUsptoDocumentBackend

# MAX_INPUT_DOCS is the value of MAX_INPUT_DOCS environment variable or 20
MAX_INPUT_DOCS = int(os.environ.get("MAX_INPUT_DOCS", 2))

# SCRATCH_DIR is the value of SCRATCH_DIR environment variable or "scratch"
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", "scratch")

# Acceptable input formats with extensions and their corresponding FormatOption
# for example: InputFormat.PDF (["pdf", "PDF"]) -> PdfFormatOption
ALLOWED_INPUT_FORMATS = {
    InputFormat.PDF: [ "pdf", "PDF"],
    InputFormat.DOCX: ["docx", "DOCX"],
    InputFormat.PPTX: ["pptx", "PPTX"],
    InputFormat.MD: ["md","MD"],
    InputFormat.HTML: ["html","HTML"],
}

# Set the log level for the 'docling.utils' logger to ERROR
logging.getLogger('docling').setLevel(logging.ERROR)

# Create a logger for this module
_log = logging.getLogger(__name__)

def export_documents(
    conv_results: list[ConversionResult],
    output_dir: Path,
):
    """
    Export the converted documents to the specified output directory.
    Args:
        conv_results (Iterable[ConversionResult]): Conversion results to export.
        output_dir (Path): Directory to save the converted documents.
    Returns:
        Tuple[int, int, int]: Counts of successful, partially successful, and failed conversions.
    """

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize counts for success, failure, and partial success
    success_count = 0
    failure_count = 0
    partial_success_count = 0

    # Iterate over the conversion results and export the documents
    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            doc_filename = conv_res.input.file.stem

            if conv_res.document:
                conv_res.document.save_as_json(
                    output_dir / f"{doc_filename}.json",
                    image_mode=ImageRefMode.PLACEHOLDER,
                )
                conv_res.document.save_as_doctags(
                    output_dir / f"{doc_filename}.doctags.txt"
                )
                conv_res.document.save_as_markdown(
                    output_dir / f"{doc_filename}.md",
                    image_mode=ImageRefMode.PLACEHOLDER,
                )
            else:
                _log.error(
                    f"Document {conv_res.input.file} was converted but no document was created."
                )
                with (output_dir / f"{doc_filename}.json").open("w") as fp:
                    json.dump(conv_res.to_dict(), fp, indent=2)

        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            _log.error(
                f"Document {conv_res.input.file} was partially converted with the following errors:"
            )
            for item in conv_res.errors:
                _log.error(f"\t{item.error_message}")
            partial_success_count += 1
        else:
            _log.fatal(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1

    _log.debug(
        f"Processed {success_count + partial_success_count + failure_count} docs, "
        f"of which {failure_count} failed "
        f"and {partial_success_count} were partially converted."
    )
    return success_count, partial_success_count, failure_count

def convert_batch(
    input_doc_paths: list[Path],
    output_dir: Path,
    pipeline_options: PdfPipelineOptions,
    converter: DocumentConverter,
):
    """
    Convert a batch of documents using the provided converter and pipeline options.
    Args:
        input_doc_paths (Iterable[Path]): Paths to the input documents.
        output_dir (Path): Directory to save the converted documents.
        pipeline_options (PdfPipelineOptions): Options for the PDF pipeline.
        converter (DocumentConverter): The document converter instance.
    Returns:
        Tuple[int, int, int]: Counts of successful, partially successful, and failed conversions.
    """
    # Raise value errors if the input document paths are empty or invalid
    if not input_doc_paths:
        raise ValueError("No input document paths provided.")
    if not all(path.exists() for path in input_doc_paths):
        raise ValueError("Some input document paths do not exist.")
    if not all(path.is_file() for path in input_doc_paths):
        raise ValueError("Some input document paths are not files.")
    
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Raise value errors if pipeline options are invalid
    if not isinstance(pipeline_options, PdfPipelineOptions):
        raise ValueError("Invalid pipeline options provided.")
    if not isinstance(converter, DocumentConverter):
        raise ValueError("Invalid document converter provided.")

    # Convert the documents
    conv_results = converter.convert_all(
        input_doc_paths,
        raises_on_error=False,  # to let conversion run through all and examine results at the end
    )
    success_count, partial_success_count, failure_count = export_documents(
        conv_results, output_dir=output_dir
    )

    return success_count, partial_success_count, failure_count

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
def docling_convert(
    input_doc_paths: list[str],
    output_dir: Path
) -> tuple[list[str], list[str], list[str]]:
    # Log input document paths
    _log.info(f"Input document paths: {', '.join(input_doc_paths)}")
    # Log output directory
    _log.info(f"Output directory: {output_dir}")
    
    # Raise value errors if the input document paths are empty or invalid
    if not input_doc_paths:
        raise ValueError("No input document paths provided.")

    # Check if input_doc_paths is a list of strings
    if not all(isinstance(path, str) for path in input_doc_paths):
        raise ValueError("Some input document paths are not strings.")

    # Check if all input document paths exist and separate them non existing paths and existing paths
    input_doc_paths = [Path(path) for path in input_doc_paths]
    non_existing_paths = [path for path in input_doc_paths if not path.exists()]
    existing_paths = [path for path in input_doc_paths if path.exists()]

    # Log the non-existing paths
    if non_existing_paths:
        _log.warning(
            f"The following input document paths do not exist: {', '.join(map(str, non_existing_paths))}"
        )

    # Convert existing_paths to a list of Path objects
    existing_paths = [Path(path) for path in existing_paths]

    # Check if all existing paths have supported extensions and separate them into valid and invalid extensions
    valid_extensions = [
        ext for fmt, exts in ALLOWED_INPUT_FORMATS.items() for ext in exts
    ]
    invalid_extensions = [
        path for path in existing_paths if path.suffix[1:] not in valid_extensions
    ]
    if invalid_extensions:
        # Log the invalid extensions
        _log.info(
            f"The following input document paths have unsupported extensions: {', '.join(map(str, invalid_extensions))}"
        )

    # Accept only the paths with valid extensions
    accepted_paths = [
        path for path in existing_paths if path.suffix[1:] in valid_extensions
    ]

    # Extract the set of extensions from the accepted paths
    extensions = set(path.suffix[1:] for path in accepted_paths)

    # Create batches of MAX_INPUT_DOCS documents out of the input existing_paths
    batches = [
        accepted_paths[i : i + MAX_INPUT_DOCS] for i in range(0, len(accepted_paths), MAX_INPUT_DOCS)
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
        InputFormat.XML_PUBMED: FormatOption(
            pipeline_cls=SimplePipeline, backend=PubMedDocumentBackend
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
            if conv_res.status == ConversionStatus.SUCCESS:
                succesfully_converted_documents.append(conv_res.input.file)
            elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
                partially_converted_documents.append(conv_res.input.file)
            else:
                failed_documents.append(conv_res.input.file)

        # Export the documents in the batch
        success_count, partial_success_count, failure_count = export_documents(
            conv_results, output_dir=output_dir
        )
        # Log the conversion results
        _log.debug(
            f"Processed {success_count + partial_success_count + failure_count} docs in batch, "
            f"of which {failure_count} failed "
            f"and {partial_success_count} were partially converted."
        )
    # Log the non-existing paths
    if non_existing_paths:
        _log.warning(
            f"The following input document paths do not exist: {', '.join(map(str, non_existing_paths))}"
        )
    # Log the total number of documents processed
    _log.info(
        f"Processed a total of {len(input_doc_paths)} documents, "
        f"of which {len(existing_paths)} existed and {len(non_existing_paths)} did not."
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
    logging.basicConfig(level=logging.INFO)

    input_doc_paths = [
        # "./tests/data/2203.01017v2.pdf",
        # "./tests/data/2206.01062v1.pdf",
        "./tests/data/2305.03393v1.pdf",
        "./tests/data/GenAI Proof of Concept Process Guidebook.docx",
        "./tests/data/Innovate Together.pptx",
        "./tests/data/readme-test.md",
        "./tests/data/L1 Ether Application Engine.md",
        "./tests/data/Workplace Introduction.md",
        "./tests/data/non_existing_file.pdf",
        "./tests/data/invalid_extension.txt",
    ]

    # Start the timer
    start_time = time.time()

    # Convert the documents
    success, partial_success, failure = docling_convert(
        input_doc_paths,
        output_dir=Path(SCRATCH_DIR),
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