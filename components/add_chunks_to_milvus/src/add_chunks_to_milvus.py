import os
import pickle

from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModel
import torch
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

from kfp import compiler
from kfp import dsl
from kfp.dsl import Input, Output, Dataset

NAMESPACE = os.environ.get("NAMESPACE", "default")
COMPONENT_NAME = os.getenv("COMPONENT_NAME", f"s3_sync")
BASE_IMAGE = os.getenv("BASE_IMAGE", "python:3.11-slim-bullseye")
REGISTRY = os.environ.get("REGISTRY", f"image-registry.openshift-image-registry.svc:5000/{NAMESPACE}")
TAG = os.environ.get("TAG", f"latest")
TARGET_IMAGE = f"{REGISTRY}/{COMPONENT_NAME}:{TAG}"

LANGCHAIN_COMMUNITY_PIP_VERSION = "0.3.20"
PYPDF_PIP_VERSION = "5.4.0"

# Generate an embedding for a given text
def generate_embedding(text: str, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()

# Add chunks to Milvus
@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["pymilvus", "transformers", "torch", "langchain_core", "einops"]
)
def add_chunks_to_milvus(
    model_name: str,
    milvus_collection_name: str,
    chunks_input_dataset: Input[Dataset]
):
    # Collection name
    print(f"milvus_collection_name: {milvus_collection_name}")

    # Get the Mivus connection details
    milvus_host = os.environ.get('MILVUS_HOST')
    milvus_port = os.environ.get('MILVUS_PORT')
    milvus_username = os.environ.get('MILVUS_USERNAME')
    milvus_password = os.environ.get('MILVUS_PASSWORD')
    
    # Print the connection details
    print(f"milvus_host: {milvus_host}")
    print(f"milvus_port: {milvus_port}")

    # Initialize Hugging Face model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model_kwargs = {'trust_remote_code': True}
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True) # TODO: Add trust_remote_code=True to avoid warnings
    vector_size = model.config.hidden_size

    # Get the chunks from the input dataset
     # Try to load the chunks from the input dataset
    try:
        print(f"Loading chunks from {chunks_input_dataset.path}")
        with open(chunks_input_dataset.path, 'rb') as f:
            chunks = pickle.load(f)
    except Exception as e:
        print(f"Failed to load chunks: {e}")

    # Check if the variable exists and is of the correct type
    if chunks is None:
        raise ValueError("Chunks not loaded successfully.")
    
    if not isinstance(chunks, list): #  or not all(isinstance(doc, Document) for doc in chunks):
        raise TypeError("The loaded data is not a List[langchain_core.documents.Document].")

    print(f"Loaded {len(chunks)} chunks")

    # Connect to Milvus
    connections.connect("default", host=milvus_host, port=milvus_port, user=milvus_username, password=milvus_password)

    # Drop collections before creating them?
    drop_before_create = True

    # Delete the chunks collections if it already exists and drop_before_create is True
    if utility.has_collection(milvus_collection_name) and drop_before_create:
        utility.drop_collection(milvus_collection_name)

    # Define the chunks collection schema
    milvus_collection_fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_size),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=8),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="page_start", dtype=DataType.INT16),
        FieldSchema(name="page_end", dtype=DataType.INT16),
        FieldSchema(name="dossier", dtype=DataType.VARCHAR, max_length=128),
    ]
    chunks_collection_schema = CollectionSchema(milvus_collection_fields, "Schema for storing document chunks.")

    # Create a collection for chunks
    milvus_chunks_collection = None
    if utility.has_collection(milvus_collection_name):
        milvus_chunks_collection = Collection(milvus_collection_name)
    else:
        milvus_chunks_collection = Collection(milvus_collection_name, chunks_collection_schema)

    # Prepare data for insertion
    vectors = [generate_embedding(doc_chunk["text"], tokenizer, model) for doc_chunk in chunks]

    # Extract attr from the doc_chunks and store them in separate lists
    sources = [doc_chunk['source'] for doc_chunk in chunks]
    languages = [doc_chunk['language'] for doc_chunk in chunks]
    texts = [doc_chunk['text'] for doc_chunk in chunks]
    page_starts = [doc_chunk['page_start'] for doc_chunk in chunks]
    page_ends = [doc_chunk['page_end'] for doc_chunk in chunks]
    dossiers = [doc_chunk['dossier'] for doc_chunk in chunks]

    # Insert data into the collection
    entities = [
        vectors,
        sources,
        languages,
        texts,
        page_starts,
        page_ends,
        dossiers,
    ]

    # Print shape of the entities
    print(f"Entities shape: {len(entities)} x {len(entities[0])} : {len(entities[1])} : {len(entities[2])} : {len(entities[3])} : {len(entities[4])} : {len(entities[5])}")

    insert_result = milvus_chunks_collection.insert(entities)
    milvus_chunks_collection.flush()

    print(f"Inserted {len(insert_result.primary_keys)} records into Milvus.")

    # Add an index to the collection
    index_params = {
        "metric_type": "L2",
        "index_type": "HNSW",
        "params": {"efConstruction": 200, "M": 16}  # Typical values for efConstruction and M
    }
    milvus_chunks_collection.create_index(field_name="vector", index_params=index_params)

    # Load the collection
    milvus_chunks_collection.load()


if __name__ == "__main__":
    component_package_path = __file__.replace('.py', '.yaml')
    compiler.Compiler().compile(
        pipeline_func=add_chunks_to_milvus,
        package_path=component_package_path
    )
