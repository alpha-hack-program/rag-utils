#!/bin/bash

# Stop and remove the existing Milvus container
podman stop milvus-ui 1> /dev/null
podman rm milvus-ui 1> /dev/null

# Localhost IP address
# Get the IP address of the host machine in macos
host_ip=$(ipconfig getifaddr en0)

echo "Host IP: $host_ip"

# Run the Milvus UI container
podman run --name milvus-ui --rm -it -p 8000:3000  -e MILVUS_URL=${host_ip}:19530 zilliz/attu:v2.5
