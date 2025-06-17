from ast import pattern
import os
import hashlib
import logging
import time
import requests
import json
import csv

from tqdm import tqdm

from pathlib import Path

from docling_core.transforms.chunker.hierarchical_chunker import DocChunk

from kfp import compiler
from kfp import dsl

# Allowed log levels
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

# Read from environment variable
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "WARNING").upper()

# Set the log level for the 'docling.utils' logger to ERROR
logging.getLogger('docling').setLevel(logging.ERROR)

# Create a logger for this module
_log = logging.getLogger(__name__)

NAMESPACE = os.environ.get("NAMESPACE", "default")
COMPONENT_NAME = os.getenv("COMPONENT_NAME", f"s3_sync")
BASE_IMAGE = os.getenv("BASE_IMAGE", "python:3.11-slim-bullseye")
REGISTRY = os.environ.get("REGISTRY", f"image-registry.openshift-image-registry.svc:5000/{NAMESPACE}")
TAG = os.environ.get("TAG", f"latest")
TARGET_IMAGE = f"{REGISTRY}/{COMPONENT_NAME}:{TAG}"

DOCLING_CORE_VERSION = "2.31.0"
REQUESTS_VERSION = "2.32.3"

# MAX_INPUT_DOCS is the value of MAX_INPUT_DOCS environment variable or 20
MAX_INPUT_DOCS = int(os.environ.get("MAX_INPUT_DOCS", 2))

# Get the OpenAI connection details
openai_api_key = os.environ.get('OPENAI_API_KEY', '')
openai_api_model = os.environ.get('OPENAI_API_MODEL')
openai_api_embeddings_url = os.environ.get('OPENAI_API_BASE')

# Prompt template for the LLM to generate questions and answers given a context
SYSTEM_PROMPT = """
You are a helpful assistant that can generate questions given the context of a document.
Questions should self-contained and answerable based only the context provided.
Parse the array "questions" and "answers" and output them as a JSON object. 

EXAMPLE INPUT:
Generate '2' questions and answers based on this context:
Mount Everest, known locally as Sagarmatha in Nepal and Qomolangma in Tibet, is Earth's highest mountain above sea level.

EXAMPLE JSON OUTPUT:
{
    "questions": [
        {
            "question": "What is the highest mountain on Earth?",
            "answer": "Mount Everest"
        },
        {
            "question": "What is the local name for Mount Everest in Nepal?",
            "answer": "Sagarmatha"
        }
    ]
}
"""

USER_PROMPT_TEMPLATE = """
Generate '{number_of_questions}' questions and answers based on this context:
{context}
```
"""

RESPONSE_FORMAT_SCHEMA = {
  "type": "json_object",
  "schema": {
    "type": "object",
    "properties": {
      "questions": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "question": { "type": "string" },
            "answer": { "type": "string" }
          },
          "required": ["question", "answer"]
        }
      }
    },
    "required": ["questions"]
  }
}

MINIMUM_CHUNK_LENGTH_FOR_QA = 768  # Minimum length of chunk text to be considered for QA generation
MAX_TOKENS = 512  # Maximum number of tokens for the response
MAX_QUESTIONS = 5  # Maximum number of questions to generate
TEMPERATURE = 1.0  # Temperature for the model response

def check_openai_connection():
    """
    Check the connection to OpenAI API using requests.
    Args:
        openai_api_key (str): OpenAI API key.
        openai_api_embeddings_url (str): OpenAI API embeddings URL.
    Raises:
        requests.exceptions.RequestException: If the connection to OpenAI API fails.
    """
    try:
        headers = {
            "Content-Type": "application/json",
        }
        # Add authorization header if the API key is set
        if openai_api_key:
            headers["Authorization"] = f"Bearer {openai_api_key}"

        response = requests.get(
            f"{openai_api_embeddings_url}/v1/models",
            headers=headers,
        )
        response.raise_for_status()
        _log.info("Connected to OpenAI successfully.")
    except requests.exceptions.RequestException as e:
        _log.error(f"Failed to connect to OpenAI: {e}")
        raise

def compute_content_hash(content: str) -> str:
    """
    Compute hash of content for duplicate detection
    
    Args:
        content (str): Content to hash
        
    Returns:
        str: SHA-256 hash of the content
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def use_preferred_prompt_template(content: str) -> str:
    """
    Use the preferred template for the model to generate the embedding.
    
    Args:
        content (str): Content to embed.
        
    Returns:
        str: Content with the preferred template applied.
    """
    # If the OpenAI API model is not set, return the content as is
    if not openai_api_model:
        _log.warning("OpenAI API model is not set. Using content as is.")
        return content

    # If model name starts with multilingual-e5-large, use the preferred template
    if openai_api_model.startswith("multilingual-e5-large"):
        # Use the preferred template for multilingual-e5-large
        content = f"passage: {content}"
    
    return content

def response_format(model: str) -> dict[str, str] | str | None:
    """
    Determine the response format based on the model. It returns 'json_object' for deepseek models,
    'json' for granite models, 'text' for others, or None if model is None or empty.
    
    Args:
        model (str): The OpenAI model name.
        
    Returns:
        dict[str, str] | str | None: The response format for the model.
    """
    if not model:
        return None
    
    # If model contains deepseek (ignore case), return {"type": "json_object"}
    if "deepseek" in model.lower():
        _log.debug(f"Using response format JSON for DeepSeek model: {model}")
        return {"type": "json_object"}
    # If model contains mixtral or mistral (ignore case), return {"type": "json_object"}
    elif "mixtral" in model.lower() or "mistral" in model.lower():
        _log.debug(f"Using response format JSON for Mixtral/Mistral model: {model}")
        return {"type": "json_object"}
    # If model contains granite (ignore case), return None
    elif "granite" in model.lower():
        _log.debug(f"Using response format None for Granite model: {model}")
        return None
    # If model contains llama (ignore case), return json_object
    elif "llama" in model.lower():
        _log.debug(f"Using response format JSON SCHEMA for Llama model: {model}")
        return RESPONSE_FORMAT_SCHEMA
    # By default, return None for other models
    _log.debug(f"Using default response format None for model: {model}")
    return "None"

def generate_questions_and_answers(
    context: str,
    number_of_questions: int,
    prompt_template: str,
) -> str:
    """
    Query the OpenAI model for embeddings.
    """
    openai_api_model = os.getenv("OPENAI_API_MODEL")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_api_base = os.getenv("OPENAI_API_BASE")

    _log.info(f"Using OpenAI API model: {openai_api_model}")
    _log.info(f"Using OpenAI API base: {openai_api_base}")
    _log.info(f"Using OpenAI API key: {'*' * len(openai_api_key) if openai_api_key else 'Not set'}")

    # Check if the model is set
    if not openai_api_model:
        raise ValueError("Model not set. Please set the OPENAI_API_MODEL environment variable.")

    # Check if the API base is set
    if not openai_api_base:
        raise ValueError("API base not set. Please set the OPENAI_API_BASE environment variable.")

    _log.info(f"Generating questions and answers for context with {number_of_questions} questions.")
    _log.debug(f"Number of questions to generate: {number_of_questions}")

    # Use the template to create the prompt
    prompt = prompt_template.format(
        context=context,
        number_of_questions=number_of_questions,
    )

    headers = {
        "Content-Type": "application/json",
    }
    # Add authorization header if the API key is set
    if openai_api_key:
        headers["Authorization"] = f"Bearer {openai_api_key}"

    # Build the json payload for the request
    json={
        "model": openai_api_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    response_format_value = response_format(openai_api_model)
    if response_format_value is not None:
        json["response_format"] = response_format_value

    # Use the OpenAI API to get the answer using requests and the prompt
    response = requests.post(
        f"{openai_api_base}/v1/chat/completions",
        headers=headers,
        json=json,
        timeout=60,  # Set a timeout for the request
    )
    response.raise_for_status()

    # Log the response status code
    _log.debug(f"Response status code: {response.status_code}")
    # Log the response content
    _log.debug(f"Response content: {response.text}")

    data = response.json()
    answer = data["choices"][0]["message"]["content"].strip()    

    _log.info(f"Generated questions and answers: {answer}")

    return answer

def evaluate_chunk_for_qa(doc_chunk: DocChunk) -> tuple[bool, str]:
    """
    Evaluate a DocChunk to decide if it should be processed for QA generation.
    
    Args:
        doc_chunk (DocChunk): The document chunk to evaluate.
        
    Returns:
        tuple[bool, str]: A tuple containing a boolean indicating whether the chunk should be processed,
                          and a reason for the decision.
    """
    # Check if the chunk is empty
    if not doc_chunk.text.strip():
        return False, "Chunk is empty."
    
    # Check if chunk text is too short
    if len(doc_chunk.text.strip()) < MINIMUM_CHUNK_LENGTH_FOR_QA:
        return False, f"Chunk text is too short. Minimum length is {MINIMUM_CHUNK_LENGTH_FOR_QA} characters."

    # Check if the chunk has headings
    if not doc_chunk.meta.headings:
        return False, "Chunk has no headings."
    
    # If all checks pass, process the chunk
    return True, "Chunk is valid for processing."

def generate_safe_filestem(text: str) -> str:
    """
    Generate a filesystem-safe name for the text provided.
    
    Args:
        text (str): The text to generate a filesystem-safe name for.
        
    Returns:
        str: A filesystem-safe name derived from the text.

    Raises:
        ValueError: If the text is None or empty.
    """
    # Fail if model_name is None or empty
    if not text:
        raise ValueError("Model name is not set or empty.")

    # Replace any non-alphanumeric characters with underscores
    return "".join(c if c.isalnum() else "_" for c in text).lower()

def escape_for_csv_column(value: str) -> str:
    import csv, io
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
    writer.writerow([value])
    return output.getvalue().strip('\r\n')


def process_document_chunks(
    document_chunks_path: Path,
    number_of_questions: int,
    prompt_template: str,
):
    """
    Process a single document chunk directory.
    Args:
        document_chunks_path (Path): Path to the document chunk directory.
        milvus_collection (Collection): Milvus collection object.
    Raises:
        Exception: If the processing fails.
    Returns:
        None
    """
    # Validate model name and embeddings URL
    if not openai_api_model or not openai_api_embeddings_url:
        raise ValueError("Missing required environment variables for OpenAI connection to generate embeddings.")

    # Check if the input directory exists
    if not document_chunks_path.exists():
        raise ValueError(f"Input directory {document_chunks_path} does not exist.")
    # Check if the input directory is a directory
    if not document_chunks_path.is_dir():
        raise ValueError(f"Input path {document_chunks_path} is not a directory.")
    # Check if the input directory is empty
    if not any(document_chunks_path.iterdir()):
        raise ValueError(f"Input directory {document_chunks_path} is empty.")
    # Log the input directory
    _log.debug(f"Processing document chunk directory: {document_chunks_path}")
    
    # Process chunks for QA generation
    # Iterate over all the json files in the directory
    count = 0
    for file in document_chunks_path.rglob("*.json"):
        # Check if the file is a json file
        if file.suffix == ".json":
            # Load the json file
            with open(file, "r") as f:
                data = f.read()
            # Convert the json file to a DocChunk object
            doc_chunk = DocChunk.model_validate_json(data)

            # Evaluate chunk to decide if it should be processed
            process_chunk, reason = evaluate_chunk_for_qa(doc_chunk)
            if not process_chunk:
                _log.debug(f"Skipping chunk {file} due to: {reason}")
                continue

            # Contextualize the chunk by adding the headings joined by spaces
            joined_headings = "\n".join(doc_chunk.meta.headings) if doc_chunk.meta.headings else ""
            contextualized_content = f"{joined_headings}\n{doc_chunk.text}" if joined_headings else doc_chunk.text

            # Compute the content hash
            content_hash = compute_content_hash(contextualized_content)

            # QA filename to be generated
            model_name = generate_safe_filestem(openai_api_model)
            qa_filename = f"{file.stem}_{content_hash}_{model_name}_qa.csv"

            # Check if the file already exists
            qa_file_path = document_chunks_path / qa_filename
            if qa_file_path.exists():
                _log.debug(f"QA file already exists: {qa_file_path}. Skipping generation.")
                continue

            # Get the questions and answers from the doc_chunk
            json_string = generate_questions_and_answers(
                context=contextualized_content,
                number_of_questions=number_of_questions,
                prompt_template=prompt_template,
            )

            # Write the questions and answers to a CSV file using csv module
            with open(qa_file_path, "w", newline='', encoding='utf-8') as qa_file:
                writer = csv.writer(qa_file, quoting=csv.QUOTE_MINIMAL)

                # Write the header
                writer.writerow(["document_filename", "chunk_filename", "model", "timestamp", "question", "answer", "context"])

                # Parse the JSON string
                questions_and_answers = json.loads(json_string)

                for question_answer in questions_and_answers.get("questions", []):
                    if doc_chunk.meta.origin and doc_chunk.meta.origin.filename:
                        document_filename = doc_chunk.meta.origin.filename
                    else:
                        _log.warning(f"Origin filename not available for chunk {file}. Using chunk file name instead.")
                        document_filename = file.stem

                    chunk_filename = f"{file.stem}.json"
                    model = openai_api_model or "unknown"
                    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
                    question = question_answer.get("question", "")
                    answer = question_answer.get("answer", "")
                    context = contextualized_content

                    # Write the row
                    writer.writerow([document_filename, chunk_filename, model, timestamp, question, answer, context])

            _log.debug(f"Generated QA file: {qa_file_path}")

            # Count the number of chunks processed
            count += 1
            
    # If no chunks were processed, return
    if count == 0:
        # Log the error
        _log.debug(f"No chunks were processed in {document_chunks_path}.")
        return

    _log.info(f"QA files generated: {count}")

def process_batch_of_document_chunks(
    batch: list[Path],
    number_of_questions: int,
    prompt_template: str,
    raises_on_error: bool = True,
) -> tuple[list[Path], list[Path]]:
    """
    Process a batch of dirs containing document chunks.
    Args:
        batch (list[Path]): List of document paths to process.
        raises_on_error (bool): Whether to raise an error on failure.
    Returns:
        Tuple[list[Path], list[Path]]: Lists of successfully processed and failed chunk dirs.
    """
    # Initialize lists for successful and failed conversions
    sucesses = []
    failures = []

    # Iterate over the batch of chunk dirs
    for document_chunks_path in batch:
        try:
            # Process the document chunks
            process_document_chunks(
                document_chunks_path=document_chunks_path,
                number_of_questions=number_of_questions,
                prompt_template=prompt_template,
            )
            # If successful, add to the list of successes
            sucesses.append(document_chunks_path)
        except Exception as e:
            _log.error(f"Failed to process {document_chunks_path}: {e}")
            if raises_on_error:
                raise
            failures.append(document_chunks_path)

    return sucesses, failures

def merge_all_csv_files_for_model(input_dir: Path, output_file: Path) -> None:
    """
    Merge all CSV files in the input directory into a single CSV file using the csv module.
    This function will create the output file if it does not exist, and overwrite it if it does.
    Args:
        input_dir (Path): Path to the input directory containing CSV files.
        output_file (Path): Path to the output CSV file.
    """
    if not input_dir.exists():
        raise ValueError(f"Input directory {input_dir} does not exist.")

    if not openai_api_model:
        raise ValueError("OpenAI API model is not set. Please set the OPENAI_API_MODEL environment variable.")

    # Merging these files *_{model_name}_qa.csv
    pattern = f"*_{generate_safe_filestem(openai_api_model)}_qa.csv"
    csv_files = list(input_dir.rglob(pattern))
    if not csv_files:
        _log.warning(f"No CSV files found in {input_dir}.")
        return

    # Clean up the output file if it exists
    if output_file.exists():
        _log.info(f"Output file {output_file} already exists. Deleting it.")
        output_file.unlink()

    # Create the output file
    with output_file.open("w", newline='', encoding='utf-8') as out_f:
        writer = None

        for idx, file_path in enumerate(csv_files):
            with file_path.open("r", newline='', encoding='utf-8') as in_f:
                reader = csv.reader(in_f)
                header = next(reader, None)

                if header is None:
                    _log.warning(f"Skipping empty file: {file_path}")
                    continue

                if writer is None:
                    writer = csv.writer(out_f)
                    writer.writerow(header)
                elif idx == 0:
                    writer.writerow(header)  # write header from the first file only

                for row in reader:
                    writer.writerow(row)

    _log.info(f"Merged {len(csv_files)} CSV files into {output_file}.")

def delete_all_csv_files(input_dir: Path) -> None:
    """
    Delete all CSV files in the input directory.
    Args:
        input_dir (Path): Path to the input directory containing CSV files.
    """
    if not input_dir.exists():
        raise ValueError(f"Input directory {input_dir} does not exist.")

    csv_files = list(input_dir.rglob("*.csv"))
    if not csv_files:
        _log.warning(f"No CSV files found in {input_dir}.")
        return

    for file_path in csv_files:
        try:
            file_path.unlink()
            _log.info(f"Deleted CSV file: {file_path}")
        except Exception as e:
            _log.error(f"Failed to delete {file_path}: {e}")

def _generate_qa_per_chunk(
    input_dir: Path,
    output_dir: Path,
    number_of_questions: int,
    cleanup: bool,
    merge_csv: bool = False,
    merged_csv_filestem_prefix: str = "merged_qa",
    prompt_template: str = USER_PROMPT_TEMPLATE,
) -> tuple[list[Path], list[Path]]:
    """
    Convert documents using Docling.
    Args:
        input_dir (Path): Path to the input directory.
        milvus_collection_name (str): Name of the Milvus collection.
    Returns:
        Tuple[list[Path], list[Path]]: Lists of successfully added, and failed.
    """

    # Validate input parameters
    if not input_dir:
        raise ValueError("Input directory is not set.")

    # Validate number of questions
    if not number_of_questions or number_of_questions <= 0:
        raise ValueError("Number of questions must be a positive integer.")
    if number_of_questions > 5:
        raise ValueError("Number of questions must be less than or equal to 5.")

    # Validate OpenAI connection parameters
    if not openai_api_model or not openai_api_embeddings_url:
        raise ValueError("Missing required environment variables for OpenAI connection to generate embeddings.")

    # Fail if merged_csv_filestem_prefix is empty
    if not merged_csv_filestem_prefix.strip():
        _log.warning("Merged CSV file stem prefix is not set.")
        raise ValueError("Merged CSV file stem prefix is not set.")

    # Log input directory
    _log.info(f"Input directory: {input_dir}")

    # Log OpenAI connection details
    _log.info(f"Connecting to OpenAI API with model: {openai_api_model}")
    _log.info(f"Using OpenAI API key: {'*' * len(openai_api_key) if openai_api_key else 'Not set'}")
    _log.info(f"Using OpenAI API embeddings URL: {openai_api_embeddings_url}")

    # Fail if openai_api_model or openai_api_embeddings_url are not set
    if not all([ openai_api_model, openai_api_embeddings_url]):
        raise ValueError("Missing required environment variables for OpenAI connection to generate embeddings.")
    
    # Fail if the input directory is not set or does not exist
    if not input_dir:
        raise ValueError("Input directory is not set.")
    if not input_dir.exists():
        raise ValueError(f"Input directory {input_dir} does not exist.")
    
    # Test the connection to OpenAI using requests
    check_openai_connection()

    # Iterate over all the directories in the input directory looking for chunk directories "*.chunks"
    # then save the paths in chunk_dir_paths
    chunk_dir_paths = []
    for file in input_dir.rglob("*"):
        # Check if the file is a directory and ends with ".chunks"
        if file.is_dir() and file.name.endswith(".chunks"):
            chunk_dir_paths.append(file)
        

    # Log the number of input documents
    _log.info(f"Found {len(chunk_dir_paths)} chunk directories in the input dir.")
    # Log the input document paths
    _log.debug(f"Input document paths: {', '.join(map(str, chunk_dir_paths))}")

    # Create batches of MAX_INPUT_DOCS documents out of the input existing_paths
    batches = [
        chunk_dir_paths[i : i + MAX_INPUT_DOCS] for i in range(0, len(chunk_dir_paths), MAX_INPUT_DOCS)
    ]

    # Log the number of batches created
    _log.info(  
        f"Created {len(batches)} batches of chunk dirs, each with a maximum of {MAX_INPUT_DOCS} dirs."
    )    

    # We're ready to process the batches of chunk directories
    _log.info("Processing batches of chunk directories...")

    # if we have to cleanup, delete all CSV files in the input directory
    if cleanup:
        _log.info("Cleaning up CSV files in the input directory...")
        delete_all_csv_files(input_dir)

    # List of successfully inserted sets of chunks
    succesfully_added_chunk_sets = []
    # List of failed sets of chunks
    failed_chunk_sets = []

    # Convert the documents in batches, use tqdm to show progress
    for batch in tqdm(batches):
        # Process each batch of chunk directories
        sucesses, failures = process_batch_of_document_chunks(
            batch=batch,
            number_of_questions=number_of_questions,
            prompt_template=prompt_template,
            raises_on_error=False,
        )
        # Add the successfully processed chunk directories to the list
        succesfully_added_chunk_sets.extend(sucesses)
        # Add the failed chunk directories to the list
        failed_chunk_sets.extend(failures)
        # Log the number of successfully processed chunk directories
        _log.debug(
            f"Successfully processed {len(sucesses)} chunk directories, "
            f"failed to process {len(failures)} chunk directories."
        )

    # Log the total number of documents processed
    _log.info(
        f"Processed a total of {len(chunk_dir_paths)} documents"
    )
    # Return the counts of successful, partial success, and failure
    success_count = len(succesfully_added_chunk_sets)
    failure_count = len(failed_chunk_sets)
    # Log the total number of documents processed
    _log.info(
        f"Processed {success_count  + failure_count} docs, "
        f"of which {failure_count} failed"
    )

    # If merge_csv is True, merge all CSV files in the input directory into a single CSV file
    if merge_csv:
        timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        merged_filestem = generate_safe_filestem(f"{merged_csv_filestem_prefix}_{openai_api_model}_{timestamp}")
        merged_csv_file = output_dir / f"{merged_filestem}.csv"
        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        # Log the merging of CSV files
        _log.info(f"Merging all CSV files in {input_dir} into {merged_csv_file}.")
        merge_all_csv_files_for_model(input_dir=input_dir, output_file=merged_csv_file)
        _log.info(f"Merged CSV file created at: {merged_csv_file}")
    else:
        _log.info("Skipping merging of CSV files as merge_csv is set to False.")

    # Return the lists of successful, partial success, and failure conversions
    return (
        succesfully_added_chunk_sets,
        failed_chunk_sets,
    )

@dsl.component(
    base_image=BASE_IMAGE,
    target_image=TARGET_IMAGE,
    packages_to_install=[
        f"docling-core=={DOCLING_CORE_VERSION}",
        f"requests=={REQUESTS_VERSION}",
    ]
)
def generate_qa_per_chunk(
    root_mount_path: str,
    input_dir_name: str,
    output_dir_name: str,
    number_of_questions: int,
    cleanup: bool = False,
    merge_csv: bool = True,
    merged_csv_filestem_prefix: str = "merged_qa",
) -> str:
    """
    Generate questions and answers for each chunk in the input directory and save them as CSV files.
    Args:
        root_mount_path (str): Root mount path where the input directory is located.
        input_dir_name (str): Name of the input directory containing chunk directories.
        number_of_questions (int): Number of questions to generate for each chunk.
    Returns:
        str: JSON string containing lists of successfully processed and failed chunk directories.
    """
    # Validate input parameters
    if not root_mount_path:
        raise ValueError("Root mount path is not set.")
    if not input_dir_name:
        raise ValueError("Input directory name is not set.")
    if not number_of_questions or number_of_questions <= 0:
        raise ValueError("Number of questions must be a positive integer.")
    if number_of_questions > 5:
        raise ValueError("Number of questions must be less than or equal to 5.")
    
    # Construct the input directory path
    input_dir = Path(root_mount_path) / input_dir_name

    # Construct the output directory path
    output_dir = Path(root_mount_path) / output_dir_name
    
    # Check if the input directory exists
    if not input_dir.exists():
        raise ValueError(f"Input directory {input_dir} does not exist.")
    
    # Check if the input directory is a directory
    if not input_dir.is_dir():
        raise ValueError(f"Input path {input_dir} is not a directory.")

    # Log input directory
    _log.info(f"Input directory: {input_dir}")

    # Log the number of questions to generate
    _log.info(f"Number of questions to generate: {number_of_questions}")
    
    # Generate questions and answers for each chunk in the input directory
    success, failure = _generate_qa_per_chunk(
        input_dir=input_dir,
        output_dir=output_dir,
        number_of_questions=number_of_questions,
        cleanup=cleanup,
        merge_csv=merge_csv,
        merged_csv_filestem_prefix=merged_csv_filestem_prefix,
    )

    # Log the number of successfully added and failed chunk directories
    _log.info(
        f"Successfully added {len(success)} chunk directories, "
        f"failed to add {len(failure)} chunk directories."
    )

    # return the lists as a json string, converting Path objects to strings
    _log.info("Returning success and failure lists as JSON.")
    return json.dumps({
        "success": [str(path) for path in success],
        "failure": [str(path) for path in failure],
    })


if __name__ == "__main__":
    component_package_path = __file__.replace('.py', '.yaml')
    compiler.Compiler().compile(
        pipeline_func=generate_qa_per_chunk, # type: ignore
        package_path=component_package_path
    )
