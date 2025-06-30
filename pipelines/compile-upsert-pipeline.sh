#!/bin/bash

# Expect one argument, the name of the pipeline to compile
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 my-pipeline"
    exit 1
fi

# Pipeline name
PIPELINE=$1

echo "Compiling pipeline ${PIPELINE}"

# Compoments are one level up
COMPONENTS_DIR=$(dirname "$(pwd)")/components

# If COMPONENTS_DIR does not exist, print error and exit
if [ ! -d "${COMPONENTS_DIR}" ]; then
  echo "Error: COMPONENTS_DIR ${COMPONENTS_DIR} does not exist."
  exit 1
fi

# Export BASE_IMAGE, REGISTRY, TAG from the .env file in COMPONENTS_DIR
if [ ! -f "${COMPONENTS_DIR}/.env" ]; then
  echo "Error: .env file not found in ${COMPONENTS_DIR}. Please create it with BASE_IMAGE, REGISTRY, and TAG variables."
  exit 1
fi
# Load the base environment variables
. ${COMPONENTS_DIR}/.env

# Export the variables so Python can see them
export BASE_IMAGE
export REGISTRY  
export TAG

# Add ${COMPONENTS_DIR} to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${COMPONENTS_DIR}"

# Add all ${COMPONENTS_DIR}/${COMPONENT}/src/ to PYTHONPATH
for COMPONENT in $(ls ${COMPONENTS_DIR}); do
  if [ -d "${COMPONENTS_DIR}/${COMPONENT}/src" ]; then
    export PYTHONPATH="${PYTHONPATH}:${COMPONENTS_DIR}/${COMPONENT}/src"
  fi
done

# Echo the PYTHONPATH
echo "PYTHONPATH: ${PYTHONPATH}"

# Check if oc command is available
if ! command -v oc &> /dev/null; then
  echo "WARNING: oc command not found. Please install OpenShift CLI (oc) and ensure it is in your PATH."
  echo "Trying to compile and upsert pipeline ${PIPELINE} directly."
  python ${PIPELINE}
  # Exit with success
  exit 0
fi

TOKEN=$(oc whoami -t)

# If TOKEN is empty print error and exit
if [ -z "$TOKEN" ]; then
  echo "Error: No token found. Please login to OpenShift using 'oc login' command."
  echo "Compile only mode."

  python ${PIPELINE}
  # Exit with success
  exit 0
fi

DATA_SCIENCE_PROJECT_NAMESPACE=$(oc project --short)

# If DATA_SCIENCE_PROJECT_NAMESPACE is empty print error and exit
if [ -z "$DATA_SCIENCE_PROJECT_NAMESPACE" ]; then
  echo "Error: No namespace found. Please set the namespace in bootstrap/.env file."
  exit 1
fi

DSPA_HOST=$(oc get route ds-pipeline-dspa -n ${DATA_SCIENCE_PROJECT_NAMESPACE} -o jsonpath='{.spec.host}')

echo "DSPA_HOST: ${DSPA_HOST}"

# If DSPA_HOST is empty print error and exit
if [ -z "${DSPA_HOST}" ]; then
  echo "Error: No host found for ds-pipeline-dspa. Please check if the deployment is successful."
  exit 1
fi

python ${PIPELINE} ${TOKEN} "https://${DSPA_HOST}"



