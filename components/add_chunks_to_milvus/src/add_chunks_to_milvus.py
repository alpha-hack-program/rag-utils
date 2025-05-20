import os
import hashlib
import logging
import time
import requests
import json

from datetime import datetime

from tqdm import tqdm

from pathlib import Path

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

from docling_core.transforms.chunker.hierarchical_chunker import DocChunk

from kfp import compiler
from kfp import dsl
from kfp.dsl import Input, Output, Dataset

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

LANGCHAIN_COMMUNITY_PIP_VERSION = "0.3.20"
PYPDF_PIP_VERSION = "5.4.0"

# MAX_INPUT_DOCS is the value of MAX_INPUT_DOCS environment variable or 20
MAX_INPUT_DOCS = int(os.environ.get("MAX_INPUT_DOCS", 2))

# Get the Mivus connection details
milvus_database = os.environ.get('MILVUS_DATABASE')
milvus_host = os.environ.get('MILVUS_HOST')
milvus_port = os.environ.get('MILVUS_PORT')
milvus_user = os.environ.get('MILVUS_USER')
milvus_password = os.environ.get('MILVUS_PASSWORD')

# Get the OpenAI connection details
openai_api_key = os.environ.get('OPENAI_API_KEY')
openai_api_model = os.environ.get('OPENAI_API_MODEL')
openai_api_embeddings_url = os.environ.get('OPENAI_API_BASE')

def query_milvus(
    collection_name: str,
    query_text: str,
    top_k: int = 5,
    params: dict = None,
    timeout: int = 30,
) -> list:
    """
    Perform a vector search on a Milvus collection.
    """
    # Connect to Milvus
    connections.connect(
        alias="default",
        host=milvus_host,
        port=milvus_port,
        user=milvus_user,
        password=milvus_password,
        db_name=milvus_database,
    )

    if not utility.has_collection(collection_name):
        raise ValueError(f"Collection {collection_name} does not exist.")

    collection = Collection(collection_name)
    collection.load()

    # Get query embedding
    embedding = get_embedding(query_text)

    if params is None:
        params = {
            "metric_type": "COSINE",
            "params": {"ef": 64},  # For HNSW, optional depending on your index
        }

    # Search
    results = collection.search(
        data=[embedding],  # must be a list of vectors
        anns_field="vector",
        param=params,
        limit=top_k,
        expr=None,  # You can add filtering expressions here if needed
        output_fields=["id", "content", "source"],
        timeout=timeout,
    )

    # Extract the top_k results
    hits = results[0]  # one query, so results[0] is the list of hits
    return [
        {
            "id": hit.id,
            "distance": hit.distance,
            "content": hit.entity.get("content"),
            "source": hit.entity.get("source"),
        }
        for hit in hits
    ]


def check_openai_embeddings_connection():
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
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "{openai_api_model}",
            "input": "Hello, world!",
        }
        response = requests.post(
            f"{openai_api_embeddings_url}/v1/embeddings",
            headers=headers,
            json=data,
        )
        response.raise_for_status()
        _log.info("Connected to OpenAI successfully.")
    except requests.exceptions.RequestException as e:
        _log.error(f"Failed to connect to OpenAI: {e}")
        raise

def check_milvus_connection():
    """
    Check the connection to Milvus using pymilvus.
    Args:
        milvus_database (str): Milvus database name.
        milvus_host (str): Milvus host.
        milvus_port (str): Milvus port.
        milvus_user (str): Milvus user.
        milvus_password (str): Milvus password.
    Raises:
        Exception: If the connection to Milvus fails.
    """
    try:
        connections.connect(
            alias="default",
            db_name=milvus_database,
            host=milvus_host,
            port=milvus_port,
            user=milvus_user,
            password=milvus_password,
        )
        _log.info("Connected to Milvus successfully.")
    except Exception as e:
        _log.error(f"Failed to connect to Milvus: {e}")
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

def get_embedding(
    content: str,
):
    """
    Get embedding for the content using OpenAI API.
    
    Args:
        content (str): Content to embed.
        
    Returns:
        list: Embedding vector.

    Raises:
        requests.exceptions.RequestException: If the connection to OpenAI API fails.
    """
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": openai_api_model,
        "input": content,
    }
    response = requests.post(
        f"{openai_api_embeddings_url}/v1/embeddings",
        headers=headers,
        json=data,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def process_document_chunks(
    document_chunks_path: Path,
    milvus_collection: Collection,
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
    
    # Process chunks for insertion
    embeddings = []
    sources = []
    contents = []
    content_hashes = []
    page_numbers_min = []
    page_numbers_max = []
    origins = []
    headings = []
    captions = []
    doc_items = []
    timestamps = []

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

            # Contextualize the chunk by adding the headings joined by \n
            joined_headings = "\n".join(doc_chunk.meta.headings) if doc_chunk.meta.headings else ""
            contextualized_content = f"{joined_headings}\n{doc_chunk.text}" if joined_headings else doc_chunk.text

            # Compute the content hash
            content_hash = compute_content_hash(contextualized_content)

            # Check if the content hash already exists in the collection
            existing_hash = milvus_collection.query(
                expr=f"content_hash == '{content_hash}'",
                output_fields=["id"],
                limit=1,
            )
            if existing_hash:
                _log.debug(f"Duplicate content hash found: {content_hash}. Skipping insertion.")
                continue

            # Get the embedding
            embedding = get_embedding(
                contextualized_content,
            )

            # Page numbers as the min and max of doc_item[i].prov.page_no
            # page_number_min = min(doc_chunk.meta.doc_items, key=lambda x: x.prov.page_no).prov.page_no
            # page_number_max = max(doc_chunk.meta.doc_items, key=lambda x: x.prov.page_no).prov.page_no
            page_number_min = min(
                (p.page_no for item in doc_chunk.meta.doc_items for p in item.prov),
                default=0
            )
            page_number_max = max(
                (p.page_no for item in doc_chunk.meta.doc_items for p in item.prov),
                default=0
            )


            # Generate a timestamp from the current time
            timestamp = datetime.now().isoformat()

            # Add embedding, source, content, content_hash, page number, origin, headings, captions, doc_items, timestamp to the lists
            embeddings.append(embedding)
            sources.append(doc_chunk.meta.origin.filename)
            contents.append(contextualized_content)
            content_hashes.append(content_hash)
            page_numbers_min.append(page_number_min)
            page_numbers_max.append(page_number_max)
            origins.append(doc_chunk.meta.origin.model_dump())
            headings.append(joined_headings)
            captions.append("\n".join(doc_chunk.meta.captions) if doc_chunk.meta.captions else "")
            # doc_items.append(doc_chunk.meta.doc_items)
            # Convert each item in the list to a dict
            doc_items_serialized = [item.model_dump() for item in doc_chunk.meta.doc_items]
            doc_items.append(doc_items_serialized)  # This is now a list of dicts
            
            timestamps.append(timestamp)

            # Count the number of chunks processed
            count += 1
            
    # If no chunks were processed, return
    if count == 0:
        # Log the error
        _log.debug(f"No chunks were processed in {document_chunks_path}.")
        return

    # Insert data into the collection
    entities = [
        embeddings,
        sources,
        contents,
        content_hashes,
        page_numbers_min,
        page_numbers_max,
        origins,
        headings,
        captions,
        doc_items,
        timestamps,
    ]

    insert_result = milvus_collection.insert(entities)
    milvus_collection.flush()

    print(f"Inserted {len(insert_result.primary_keys)} records into Milvus.")

def process_batch_of_document_chunks(
    batch: list[Path],
    milvus_collection: Collection,
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
                milvus_collection=milvus_collection,
            )
            # If successful, add to the list of successes
            sucesses.append(document_chunks_path)
        except Exception as e:
            _log.error(f"Failed to process {document_chunks_path}: {e}")
            if raises_on_error:
                raise
            failures.append(document_chunks_path)

    return sucesses, failures

def init(
    milvus_collection_name: str,
    embedding_dim: int,
):
    """
    Initialize Milvus connection and ensure collection exists
    
    Returns:
        Collection: Milvus collection object
    """
    try:
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=milvus_host,
            port=milvus_port,
            user=milvus_user,
            password=milvus_password,
            db_name=milvus_database,
        )
        _log.info(f"Connected to Milvus server at {milvus_host}:{milvus_port}")
        
        # Check if collection exists
        if not utility.has_collection(milvus_collection_name):
            # Create collection
            collection = _create_collection(
                milvus_collection_name=milvus_collection_name,
                embedding_dim=embedding_dim,
            )
            _log.info(f"Created new collection: {milvus_collection_name}")
        else:
            collection = Collection(milvus_collection_name)
            _log.info(f"Using existing collection: {milvus_collection_name}")
            
            # Ensure vector index exists
            vector_index_exists = any(idx.field_name == "vector" for idx in collection.indexes)
            if not vector_index_exists:
                _log.warning("Vector index missing on existing collection. Creating index...")
                _create_indexes(collection)  # This will add the vector index

                # Wait for index
                _log.info(f"Waiting for vector index on existing collection '{milvus_collection_name}'...")
                wait_for_index(collection, "vector_index")

        # Load collection
        collection.load()
        return collection
        
    except Exception as e:
        _log.error(f"Initialization failed: {e}")
        raise


def _create_collection(
        milvus_collection_name: str,
        embedding_dim: int,
) -> Collection:
    """
    Create a new collection with the DocChunk schema
    
    Returns:
        Collection: Newly created collection
    """
    # Define collection schema
    id = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    vector = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    source = FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255)
    content = FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535)
    content_hash = FieldSchema(name="content_hash", dtype=DataType.VARCHAR, max_length=64)  # For duplicate detection
    page_number_min = FieldSchema(name="page_number_min", dtype=DataType.INT16)
    page_number_max = FieldSchema(name="page_number_max", dtype=DataType.INT16)
    origin = FieldSchema(name="origin", dtype=DataType.JSON, max_length=65535)
    headings = FieldSchema(name="headings", dtype=DataType.VARCHAR, max_length=1024)
    captions = FieldSchema(name="captions", dtype=DataType.VARCHAR, max_length=255)
    doc_items = FieldSchema(name="doc_items", dtype=DataType.JSON, max_length=65535)
    timestamp = FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=30)

    # Create collection schema
    schema = CollectionSchema(
        fields=[id, vector, source, content, content_hash, page_number_min, page_number_max, origin, headings, captions, doc_items, timestamp],
        description="DocChunk collection from IBM Docling"
    )
    
    # Create collection
    collection = Collection(name=milvus_collection_name, schema=schema)
    _log.info(f"Created new collection: {milvus_collection_name}")
    
    # Create indexes
    _create_indexes(collection)

   # Wait for the index to be ready
    wait_for_index(collection, "vector_index")

    # Now load the collection
    collection.load()
    
    return collection

def wait_for_index(collection, index_name: str, sleep_seconds: int = 1):
    while True:
        try:
            collection.describe_index(index_name=index_name)
            return
        except Exception as e:
            _log.info(f"Waiting for index '{index_name}': {e}")
            time.sleep(sleep_seconds)

def _create_indexes(collection):
    """
    Create necessary indexes for efficient operations
    
    Args:
        collection: Milvus collection object
    """
    # Vector index (HNSW or IVF_FLAT, etc.)
    try:
        collection.create_index(
            field_name="vector",
            index_name="vector_index",
            index_params={
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 200}
            }
        )
    except Exception as e:
        _log.warning(f"Skipping vector index creation: {e}")
    
    try:
        collection.create_index(
            field_name="content_hash",
            index_name="hash_index",
            index_params={"index_type": "AUTOINDEX"}
        )
    except Exception as e:
        _log.warning(f"Skipping content_hash index creation: {e}")

    try:
        collection.create_index(
            field_name="source",
            index_name="source_index",
            index_params={"index_type": "AUTOINDEX"}
        )
    except Exception as e:
        _log.warning(f"Skipping source index creation: {e}")
    
    _log.info("Created indexes on the collection")

def _add_chunks_to_milvus(
    input_dir: Path,
    milvus_collection_name: str,
) -> tuple[list[str], list[str]]:
    """
    Convert documents using Docling.
    Args:
        input_dir (Path): Path to the input directory.
        milvus_collection_name (str): Name of the Milvus collection.
    Returns:
        Tuple[list[str], list[str]]: Lists of successfully added, and failed.
    """
    # Fail if any of the required environment variables are not set
    if not all([milvus_database, milvus_host, milvus_port, milvus_user, milvus_password]):
        raise ValueError("Missing required environment variables for Milvus connection.")
    if not all([openai_api_key, openai_api_model, openai_api_embeddings_url]):
        raise ValueError("Missing required environment variables for OpenAI connection to generate embeddings.")
    
    # Fail if the input directory is not set or does not exist
    if not input_dir:
        raise ValueError("Input directory is not set.")
    if not input_dir.exists():
        raise ValueError(f"Input directory {input_dir} does not exist.")

    # Log input directory
    _log.info(f"Input directory: {input_dir}")
    # Log Milvus connection details
    _log.info(f"Milvus host: {milvus_host}")
    _log.info(f"Milvus port: {milvus_port}")
    _log.info(f"Milvus user: {milvus_user}")
    _log.info(f"Milvus password: *********")
    _log.info(f"OpenAI API key: {openai_api_key}")
    _log.info(f"OpenAI API embeddings URL: {openai_api_embeddings_url}")
    
    # Check the connection to Milvus
    check_milvus_connection()

    # Test the connection to OpenAI using requests
    check_openai_embeddings_connection()
    
    # Set embedding dimension based on the model
    if openai_api_model == "/mnt/models": # NOMIC in MaaS
        embedding_dim = 768
    elif openai_api_model == "text-embedding-babbage-001":
        embedding_dim = 1024
    elif openai_api_model == "text-embedding-curie-001":
        embedding_dim = 768
    else:
        raise ValueError(f"Unsupported OpenAI model: {openai_api_model}")
    # Log the embedding dimension
    _log.info(f"Embedding dimension: {embedding_dim}")

    # Initialize
    milvus_collection = init(
        milvus_collection_name=milvus_collection_name,
        embedding_dim=embedding_dim,
    )

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

    # List of successfully inserted sets of chunks
    succesfully_added_chunk_sets = []
    # List of failed sets of chunks
    failed_chunk_sets = []

    # Convert the documents in batches, use tqdm to show progress
    for batch in tqdm(batches):
        # Process each batch of chunk directories
        sucesses, failures = process_batch_of_document_chunks(
            batch=batch,
            milvus_collection=milvus_collection,
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
    # Return the lists of successful, partial success, and failure conversions
    return (
        succesfully_added_chunk_sets,
        failed_chunk_sets,
    )

# Add chunks to Milvus
@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["pymilvus", "transformers", "torch", "langchain_core", "einops"]
)
def add_chunks_to_milvus(
    input_dir: str,
    milvus_collection_name: str,
):
    # Log input directory
    _log.info(f"Input directory: {input_dir}")

    # Log milvus collection name
    _log.info(f"Collection Name: {milvus_collection_name}")
    
    
    # Add chunks to milvus
    success, failure = _add_chunks_to_milvus(
        input_dir=input_dir,
        milvus_collection_name=milvus_collection_name,
    )

    # return the lists as a json string
    return json.dumps({
        "success": [str(path) for path in success],
        "failure": [str(path) for path in failure]
    })


if __name__ == "__main__":
    component_package_path = __file__.replace('.py', '.yaml')
    compiler.Compiler().compile(
        pipeline_func=add_chunks_to_milvus,
        package_path=component_package_path
    )
