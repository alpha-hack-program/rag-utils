import os
import json
import logging

from pathlib import Path

from typing import Optional

from tqdm import tqdm

from docling_core.types.doc.document import DoclingDocument
from docling_core.transforms.chunker.hybrid_chunker import (
    HybridChunker
)
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling.datamodel.base_models import InputFormat

from transformers import AutoTokenizer

from shared.rag_utils import is_chunked, mark_file_chunked

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

# Chunker settings
TOKENIZER_EMBED_MODEL_ID = os.environ.get("TOKENIZER_EMBED_MODEL_ID", None) # "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER_MAX_TOKENS = int(os.environ.get("TOKENIZER_MAX_TOKENS", "200")) # 200

# Acceptable input formats with extensions and their corresponding FormatOption
# for example: InputFormat.PDF (["pdf", "PDF"]) -> PdfFormatOption
ALLOWED_INPUT_FORMATS = {
    InputFormat.JSON_DOCLING: ["json","JSON"],
}

# Allowed log levels
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

# Read from environment variable
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "WARNING").upper()

# Set the log level for the 'docling.utils' logger to ERROR
logging.getLogger('docling').setLevel(logging.ERROR)

# Create a logger for this module
_log = logging.getLogger(__name__)

def create_hybrid_chunker(
    tokenizer_embed_model_id: Optional[str] = None,
    tokenizer_max_tokens: Optional[int] = None,
    merge_peers: bool = True,
) -> HybridChunker:
    """
    Create a HybridChunker with the specified parameters.
    Args:
        tokenizer_embed_model_id (Optional[str]): The model ID of the tokenizer to use. If None, a default HybridChunker is created.
        tokenizer_max_tokens (Optional[int]): The maximum number of tokens for the tokenizer. If None, it defaults to the tokenizer's model max length.
        merge_peers (bool): Whether to merge peers.
    Returns:
        HybridChunker: A HybridChunker instance.
    """
    # Log the tokenizer embed model ID and max tokens
    _log.info(f"Using tokenizer embed model ID: {tokenizer_embed_model_id}")
    _log.info(f"Using tokenizer max tokens: {tokenizer_max_tokens}")

    # If the tokenizer_embed_model_id is None create a default HybridChunker
    if tokenizer_embed_model_id is None:
        # Create a default tokenizer
        return HybridChunker()

    # Create the tokenizer
    _log.info(f"Loading tokenizer from model ID: {tokenizer_embed_model_id}")
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_embed_model_id)
    
    # Check if the tokenizer is loaded correctly
    if _tokenizer is None:
        _log.error(f"Failed to load tokenizer from model ID: {tokenizer_embed_model_id}")
        raise ValueError(f"Failed to load tokenizer from model ID: {tokenizer_embed_model_id}")
    _log.info(f"Tokenizer loaded successfully from model ID: {tokenizer_embed_model_id}")

    # Log tokenizer max allowed tokens
    _log.info(f"Tokenizer '{tokenizer_embed_model_id}' max allowed tokens: {_tokenizer.model_max_length}")

    # If the tokenizer_max_tokens is None, set it to the tokenizer's model max length
    if tokenizer_max_tokens is None:
        tokenizer_max_tokens = _tokenizer.model_max_length
        _log.info(f"Using tokenizer max tokens: {tokenizer_max_tokens} (default from tokenizer)")
    else:
        # If the tokenizer_max_tokens is set, check if it exceeds the tokenizer's model max length and error if it does
        if tokenizer_max_tokens > _tokenizer.model_max_length:
            _log.error(
                f"Tokenizer max tokens {tokenizer_max_tokens} exceeds the tokenizer's model max length {_tokenizer.model_max_length}. "
                "This may lead to issues with the tokenizer."
            )
            raise ValueError(
                f"Tokenizer max tokens {tokenizer_max_tokens} exceeds the tokenizer's model max length {_tokenizer.model_max_length}."
            )
        else:
            _log.info(
                f"Using tokenizer max tokens: {tokenizer_max_tokens} (set by user)"
            )

    # Create a HuggingFaceTokenizer with optional max_tokens
    tokenizer = HuggingFaceTokenizer(
        tokenizer=_tokenizer,
        max_tokens=tokenizer_max_tokens,
    )

    # Create a HierarchicalChunker with the specified chunk size and merge list items
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=merge_peers,  # optional, defaults to True
    )
    
    return chunker

def chunk_batch(
    batch: list[Path],
    chunker: HybridChunker,
    input_dir: Path,
    output_dir: Path,
    raises_on_error: bool = False,
) -> tuple[dict[Path,int], list[Path]]:
    """
    Chunk a batch of documents using the specified chunker. Returns a dictionary with the chunks per document.
    The chunked documents are saved in the specified output directory.
    If an error occurs, the document is added to the failed documents list.
    Args:
        batch (list[Path]): List of document paths to chunk.
        chunker (HybridChunker): The chunker to use for chunking.
        output_dir (Path): The directory to save the chunked documents.
        raises_on_error (bool): Whether to raise an error on failure. Defaults to False.
    Returns:
        tuple[dict, list[Path]]: A tuple containing:
            - Dictionary of # of chunks per document.
            - List of failed documents.
    """
    
    # List of successfully chunked documents
    chunked_documents: dict[Path,int] = {}
    # List of failed documents
    failed_documents = []

    # Iterate over the documents in the batch
    for doc_file in batch:
        try:
            # Extract the relative path of the directory the document file is in compared to the input directory
            doc_relative_path = doc_file.relative_to(input_dir).parent

            # Skip the file if it has already been chunk
            if is_chunked(doc_file):
                _log.debug(f"File {doc_file} has already been chunked. Skipping.")
                continue

            # Create a Document object from the file
            doc = DoclingDocument.load_from_json(doc_file)

            # Log the document file path
            _log.info(f"Chunking document: {doc_file}")
            
            # Get the hash of the document
            if not doc.origin or not doc.origin.binary_hash:
                _log.error(f"Document {doc_file} does not have a valid origin or binary hash. Skipping.")
                raise ValueError(f"Document {doc_file} does not have a valid origin or binary hash.")
            doc_hash = doc.origin.binary_hash

            # Chunk the document
            chunks = chunker.chunk(doc, output_dir=output_dir)

            # Create a directory with the name of the document + hash + .chunks
            doc_name = doc_file.stem
            chunks_dir = output_dir / doc_relative_path / f"{doc_name}-{doc_hash}.chunks"
            chunks_dir.mkdir(parents=True, exist_ok=True)

            # Save the chunks in the output directory
            number_of_chunks = 0
            for i, chunk in enumerate(chunks):
                contextualized_text = chunker.contextualize(chunk=chunk)
                chunk_size = chunker.tokenizer.count_tokens(contextualized_text)
                # Log the chunk size
                _log.info(f"Chunk {i} size: {chunk_size} tokens")
                # Check if the chunk size exceeds the maximum allowed tokens for the tokenizer
                if chunk_size > TOKENIZER_MAX_TOKENS:
                    _log.warning(
                        f"Chunk {i} size {chunk_size} exceeds the maximum allowed tokens {TOKENIZER_MAX_TOKENS}. "
                        "This may lead to issues with the tokenizer."
                    )
                number_of_chunks += 1
                chunk_file = chunks_dir / f"{doc_name}_chunk_{i}.json"
                with open(chunk_file, "w") as f:
                    f.write(chunk.model_dump_json())
                chunk_with_context = chunks_dir / f"{doc_name}_chunk_{i}.ctxt"
                with open(chunk_with_context, "w") as f:
                    f.write(contextualized_text)

            # Add the number of chunks to the dictionary
            chunked_documents[doc_file] = number_of_chunks

            # Mark the document as chunked
            mark_file_chunked(doc_file)
            _log.info(f"Document {doc_file} chunked successfully. Number of chunks: {number_of_chunks}")
                
        except Exception as e:
            # Log the error and add the document to the failed list
            _log.error(f"Failed to chunk document {doc_file}: {e}")
            failed_documents.append(doc_file)
            if raises_on_error:
                raise e

    return chunked_documents, failed_documents

def _docling_chunker(
    input_dir: Path,
    output_dir: Path,
    tokenizer_embed_model_id: Optional[str] = None,
    tokenizer_max_tokens: Optional[int] = None,
    merge_peers: bool = True,
) -> tuple[dict[Path,int], list[Path]]:
    """
    Chunk documents using the Docling chunker.
    Args:
        input_dir (Path): Directory containing the documents to chunk.
        output_dir (Path): Directory to save the chunked documents.
    Returns:
        tuple[list[str], list[str]]: A tuple containing:
            - List of successfully chunked documents.
            - List of failed documents.
    """
    # Log input directory
    _log.info(f"Input directory: {input_dir}")
    # Log output directory
    _log.info(f"Output directory: {output_dir}")

    # Log chunker settingsx
    _log.info(f"Using tokenizer embed model ID: {tokenizer_embed_model_id}")
    _log.info(f"Using tokenizer max tokens: {tokenizer_max_tokens}")
    _log.info(f"Using merge peers: {merge_peers}")

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

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # List of successfully chunked documents
    succesfully_chunked_documents: dict[Path,int] = {}
    # List of failed documents
    failed_documents: list[Path] = []

    # Chunker
    chunker = create_hybrid_chunker(
        tokenizer_embed_model_id=tokenizer_embed_model_id,
        tokenizer_max_tokens=tokenizer_max_tokens,
        merge_peers=merge_peers,
    )

    # Convert the documents in batches, use tqdm to show progress
    for batch in tqdm(batches):
        # Chunk the documents in the batch
        _chunked_documents, _failed_documents = chunk_batch(
            batch=batch,
            chunker=chunker,
            input_dir=input_dir,
            output_dir=output_dir,
            raises_on_error=False,  # to let conversion run through all and examine results at the end
        )

        # Push the chunked documents to the dictionary of chunked documents
        succesfully_chunked_documents.update(_chunked_documents)

        # Append the failed documents to the list of failed documents
        failed_documents.extend(_failed_documents)
        
    # Log the total number of documents chunked
    _log.info(
        f"Chunked a total of {len(input_doc_paths)} documents"
    )
    # Return the counts of successful, partial success, and failure
    success_count = len(succesfully_chunked_documents)
    failure_count = len(failed_documents)
    # Log the total number of documents chunked and failed
    _log.info(
        f"Chunked {success_count + failure_count} docs, "
        f"of which {failure_count} failed"
    )
    # Return the lists of successful, partial success, and failure conversions
    return (
        succesfully_chunked_documents,
        failed_documents,
    )

# Function that 
@dsl.component(
    base_image=BASE_IMAGE,
    target_image=TARGET_IMAGE,
    packages_to_install=[f"docling[vlm]=={DOCLING_PIP_VERSION}", f"load_dotenv=={LOAD_DOTENV_PIP_VERSION}"]
)
def docling_chunker(
    root_mount_path: str,
    input_dir_name: str,
    output_dir_name: str,
    tokenizer_embed_model_id: Optional[str] = None,
    tokenizer_max_tokens: Optional[int] = None,
    merge_peers: bool = True,
) -> str:
    """
    Convert documents using Docling.
    Args:
        root_mount_path (str): The root mount path where the input and output directories are located.
        input_dir_name (str): The name of the input directory containing documents to chunk.
        output_dir_name (str): The name of the output directory where chunked documents will be saved.
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

    # Convert output_dir to a Path object and fail if it can't be converted
    output_dir = Path(output_dir)
    if not isinstance(output_dir, Path):
        raise ValueError(f"Output directory {output_dir} is not a valid path.")

    # Chunk the documents in the input directory
    success, failure = _docling_chunker(
        input_dir=input_dir,
        output_dir=output_dir,
        tokenizer_embed_model_id=tokenizer_embed_model_id,
        tokenizer_max_tokens=tokenizer_max_tokens,
        merge_peers=merge_peers,
    )
    
    # return the lists as a json string
    return json.dumps({
        "success": [
            {"path": str(path), "chunks": chunks} for path, chunks in success.items()
        ],
        "failure": [str(path) for path in failure]
    })
    
if __name__ == "__main__":
    # Generate and save the component YAML file
    component_package_path = __file__.replace('.py', '.yaml')

    docling_chunker.save_component_yaml(component_package_path) # type: ignore

