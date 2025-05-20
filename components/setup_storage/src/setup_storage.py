import os

from kfp import dsl

from kubernetes import client
from kubernetes.client.rest import ApiException

NAMESPACE = os.environ.get("NAMESPACE", "default")
COMPONENT_NAME = os.getenv("COMPONENT_NAME", f"setup_storage")
BASE_IMAGE = os.getenv("BASE_IMAGE", "python:3.11-slim-bullseye")
REGISTRY = os.environ.get("REGISTRY", f"image-registry.openshift-image-registry.svc:5000/{NAMESPACE}")
TAG = os.environ.get("TAG", f"latest")
TARGET_IMAGE = f"{REGISTRY}/{COMPONENT_NAME}:{TAG}"

KUBERNETES_PIP_VERSION = "23.6.0"

# This component creates a PersistentVolumeClaim (PVC) in the current namespace with the specified size, 
# access mode and storage class
@dsl.component(
    base_image=BASE_IMAGE,
    target_image=TARGET_IMAGE,
    packages_to_install=[
        f"kubernetes=={KUBERNETES_PIP_VERSION}"
    ]
)
def setup_storage(
    pvc_name: str,
    size_in_gi: int,
    access_mode: str = "ReadWriteOnce",
    storage_class: str = ""
) -> None:
    """Sets up a PersistentVolumeClaim (PVC) if it does not exist.

    Args:
        pvc_name (str): Name of the PVC to create.
        size_in_gi (int): Size of the PVC in GiB.
        storage_class (str): Storage class for the PVC. Default is an empty string.

    Raises:
        ValueError: If `size_in_gi` is less than 0.
        RuntimeError: If there's any other issue in PVC creation.
    """
    if size_in_gi < 0:
        raise ValueError("size_in_gi must be a non-negative integer.")

    print(f"Creating PVC '{pvc_name}' with size {size_in_gi}Gi, access mode '{access_mode}', and storage class '{storage_class}'.") 

    # Create configuration using the service account token and namespace
    configuration = client.Configuration()

    # Load the service account token and namespace from the mounted paths
    token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
    namespace_path = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'

    # Read the token
    with open(token_path, 'r') as token_file:
        token = token_file.read().strip()

    # Read the namespace
    with open(namespace_path, 'r') as namespace_file:
        namespace = namespace_file.read().strip()

    print(f"Token: {token} Namespace: {namespace}")

    # Configure the client
    # curl -k --cacert /var/run/secrets/kubernetes.io/serviceaccount/ca.crt \
    #   -H "Authorization: Bearer $(cat /var/run/secrets/kubernetes.io/serviceaccount/token)"  \
    #   https://kubernetes.default.svc/api/v1/namespaces/iniciativa-2/persistentvolumeclaims/images-datasets-pvc
    kubernetes_host = f"https://{os.getenv('KUBERNETES_SERVICE_HOST', 'kubernetes.default.svc')}:{os.getenv('KUBERNETES_SERVICE_PORT', '443')}"
    print(f"kubernetes_host: {kubernetes_host}")
    configuration.host = kubernetes_host
    # configuration.host = 'https://kubernetes.default.svc'
    configuration.verify_ssl = True
    configuration.ssl_ca_cert = '/var/run/secrets/kubernetes.io/serviceaccount/ca.crt'
    configuration.api_key['authorization'] = token
    configuration.api_key_prefix['authorization'] = 'Bearer'

    # Print all the configuration settings
    print("Configuration settings:")
    for attr, value in vars(configuration).items():
        print(f"{attr}: {value}")

    print("Configured Kubernetes API Host:", configuration.host)
    # Create an API client with the configuration
    api_client = client.ApiClient(configuration)
    print("API Client Host:", api_client.configuration.host)

    # Use the CoreV1 API to list PVCs
    v1 = client.CoreV1Api(api_client)

    # Check if the PVC already exists
    try:
        v1.read_namespaced_persistent_volume_claim(name=pvc_name, namespace=namespace)
        print(f"PVC '{pvc_name}' already exists.")
        return
    except ApiException as e:
        if e.status != 404:
            raise RuntimeError(f"Error checking for existing PVC: {e}")

    # Define PVC spec
    pvc_spec = client.V1PersistentVolumeClaim(
        metadata=client.V1ObjectMeta(name=pvc_name),
        spec=client.V1PersistentVolumeClaimSpec(
            access_modes=[access_mode],
            resources=client.V1ResourceRequirements(
                requests={"storage": f"{size_in_gi}Gi"}
            )
        )
    )

    # Add storage class if provided
    if storage_class:
        pvc_spec.spec.storage_class_name = storage_class

    # Attempt to create the PVC
    try:
        v1.create_namespaced_persistent_volume_claim(namespace=namespace, body=pvc_spec)
        print(f"PVC '{pvc_name}' created successfully in namespace '{namespace}'.")
    except ApiException as e:
        raise RuntimeError(f"Failed to create PVC: {e.reason}")

if __name__ == "__main__":
    # Generate and save the component YAML file
    component_package_path = __file__.replace('.py', '.yaml')

    setup_storage.save_component_yaml(component_package_path)

