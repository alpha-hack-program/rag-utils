import os
import logging
import time
import argparse
import requests

from pathlib import Path

from add_chunks_to_milvus import query_milvus

# Allowed log levels
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

# Read from environment variable
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "WARNING").upper()

# Set the log level for the 'docling.utils' logger to ERROR
logging.getLogger('docling').setLevel(logging.ERROR)

# Create a logger for this module
_log = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
You are a helpful assistant that can answer questions about the content of a document.
You will be given a question and a context. Your task is to answer the question based on the context provided.
The context is a list of sentences from a document.
Don't add any additional information to the answer nor references that are not in the context.
The question is a single sentence.

The context is as follows:
{context}

The question is as follows:
{question}
"""

def query_llm(
    query: str,
    context: str,
    prompt_template: str = PROMPT_TEMPLATE,
):
    """
    Query the OpenAI model for embeddings.
    """
    model = os.getenv("OPENAI_API_MODEL_COMPLETIONS")
    openai_api_key = os.getenv("OPENAI_API_KEY_COMPLETIONS")
    openai_api_base = os.getenv("OPENAI_API_BASE_COMPLETIONS")

    # Check if the model is set
    if not model:
        raise ValueError("Model not set. Please set the OPENAI_MODEL environment variable.")

    # Check if the API base is set
    if not openai_api_base:
        raise ValueError("API base not set. Please set the OPENAI_API_BASE environment variable.")

    # Use the template to create the prompt
    prompt = prompt_template.format(
        context=context,
        question=query,
    )

    # Use the OpenAI API to get the answer using requests and the prompt
    response = requests.post(
        f"{openai_api_base}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 512,
            "temperature": 1.0,
        },
    )
    response.raise_for_status()
    data = response.json()
    answer = data["choices"][0]["message"]["content"].strip()

    return answer

# Function that given a text runs a test query on the Milvus collection
def _test_query(query: str, collection_name: str):
    """
    Run a test query on the Milvus collection.
    """

    # Perform the query
    results = query_milvus(collection_name, query)

    _log.info(f"Results for query '{query}': {results}")

    # Check if the results are empty
    if not results:
        _log.warning(f"No results found for query '{query}'")
        return
    
    # Extract the "content" field from all results and join them into a single string
    context = "\n".join([result["content"] for result in results])

    # Query the LLM with the context and the question
    answer = query_llm( 
        query=query,
        context=context,
        prompt_template=PROMPT_TEMPLATE,
    )
    _log.info(f"\nAnswer for query '{query}': {answer}")

def main():
    # Validate and set level
    if LOG_LEVEL_STR in VALID_LOG_LEVELS:
        log_level = getattr(logging, LOG_LEVEL_STR)
    else:
        print(f"Invalid LOG_LEVEL '{LOG_LEVEL_STR}' - defaulting to WARNING")
        log_level = logging.WARNING

    logging.basicConfig(level=log_level)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Parse ...")

    parser.add_argument(
        '--collection',
        required=True,
        help='Milvus Collection Name'
    )
    
    args = parser.parse_args()


    # Check environment variables for:
    # MILVUS_HOST, MILVUS_PORT, MILVUS_USER, MILVUS_PASSWORD
    # OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_API_VERSION
    MILVUS_DATABASE = os.getenv("MILVUS_DATABASE")
    MILVUS_HOST = os.getenv("MILVUS_HOST")
    MILVUS_PORT = os.getenv("MILVUS_PORT")
    MILVUS_USERNAME = os.getenv("MILVUS_USERNAME")
    MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

    if not all([MILVUS_DATABASE, MILVUS_HOST, MILVUS_PORT, MILVUS_USERNAME, MILVUS_PASSWORD]):
        raise ValueError("Missing required environment variables for Milvus connection.")
    if not all([OPENAI_API_MODEL, OPENAI_API_BASE]):
        raise ValueError("Missing required environment variables for OpenAI connection to generate embeddings.")

    # Start the timer
    start_time = time.time()

    # Run a test query 1
    # Context:
    # - Integration with Kiali: When properly configured, you can view distributed tracing data from the Kiali console.
    # - High scalability: The distributed tracing back end is designed to have no single points of failure and to scale with the business needs.
    # - Distributed Context Propagation: Enables you to connect data from different components together to create a complete end-to-end trace.
    # - Backwards compatibility with Zipkin: Red Hat OpenShift distributed tracing has APIs that enable it to be used as a drop-in replacement for Zipkin, but Red Hat is not supporting Zipkin compatibility in this release.
    test_text = "query: What are the main features of Red Hat OpenShift distributed tracing?"
    _log.info(f"Running test query: '{test_text}'")
    _test_query(test_text, args.collection)

    # Stop the timer
    end_time = time.time() - start_time
    _log.info(f"Test query completed in {end_time:.2f} seconds.")

    # Start the timer
    start_time = time.time()
    
    # Run a test query 2
    test_text = "How does Kiali integrate distributed tracing with Jaeger?"
    _log.info(f"Running test query: '{test_text}'")
    _test_query(test_text, args.collection)

    # Stop the timer
    end_time = time.time() - start_time
    _log.info(f"Test query completed in {end_time:.2f} seconds.")
    
if __name__ == "__main__":
    main()