from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Model
from azure.ai.ml.constants import AssetTypes

import argparse
import datetime


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subscription-id", dest="subscription_id", required=True)
    parser.add_argument("--resource-group", dest="resource_group", required=True)
    parser.add_argument("--workspace", dest="workspace", required=True)
    parser.add_argument("--endpoint-name", dest="endpoint_name", default="diabetes-endpoint-l9f7ce05")
    parser.add_argument("--deployment-name", dest="deployment_name", default="blue")

    return parser.parse_args()


def get_ml_client(subscription_id: str, resource_group: str, workspace: str) -> MLClient:
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )


def ensure_endpoint(ml_client: MLClient, endpoint_name: str) -> ManagedOnlineEndpoint:
    try:
        endpoint = ml_client.online_endpoints.get(name=endpoint_name)
        print(f"Endpoint '{endpoint_name}' already exists")
        return endpoint
    except Exception as e:
        print(f"Endpoint does not exist, creating new one: {str(e)}")
        unique_suffix = datetime.datetime.now().strftime("%m%d%H%M%f")
        name = endpoint_name or f"endpoint-{unique_suffix}"

        endpoint = ManagedOnlineEndpoint(
            name=name,
            description="Online endpoint for MLflow diabetes model",
            auth_mode="key",
        )

        print(f"Creating endpoint '{name}'...")
        return ml_client.begin_create_or_update(endpoint).result()


def create_or_update_deployment(
    ml_client: MLClient,
    endpoint_name: str,
    deployment_name: str,
) -> ManagedOnlineDeployment:
    # Create a model reference from the local model directory
    # This should contain MLflow model artifacts and score.py
    model = Model(
        path="./src/model",
        type=AssetTypes.MLFLOW_MODEL,
        description="MLflow diabetes classification model",
    )

    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        instance_type="Standard_D2as_v4",
        instance_count=1,
        # Add environment variables that might be useful
        environment_variables={
            "AZUREML_MODEL_DIR": "/var/azureml-app/azureml-models"
        },
        # Increase the liveness probe initial delay to give the container more time to start
        liveness_probe={
            "failure_threshold": 3,
            "initial_delay_seconds": 120,  # Increased from default 10
            "period_seconds": 10,
            "success_threshold": 1,
            "timeout_seconds": 2,
        },
        readiness_probe={
            "failure_threshold": 3,
            "initial_delay_seconds": 60,  # Increased from default 10
            "period_seconds": 10,
            "success_threshold": 1,
            "timeout_seconds": 2,
        },
    )

    print(f"Creating or updating deployment '{deployment_name}'...")
    return ml_client.online_deployments.begin_create_or_update(deployment).result()


def set_traffic_to_deployment(ml_client: MLClient, endpoint_name: str, deployment_name: str) -> None:
    print(f"Setting traffic to deployment '{deployment_name}'...")
    endpoint = ml_client.online_endpoints.get(name=endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.begin_create_or_update(endpoint).result()


def main() -> None:
    args = parse_args()

    print("Connecting to Azure Machine Learning workspace...")
    ml_client = get_ml_client(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace=args.workspace,
    )

    print(f"Ensuring online endpoint '{args.endpoint_name}' exists...")
    endpoint = ensure_endpoint(ml_client, args.endpoint_name)
    print(f"Using endpoint: {endpoint.name}")

    print(f"Creating or updating deployment '{args.deployment_name}'...")
    deployment = create_or_update_deployment(
        ml_client=ml_client,
        endpoint_name=endpoint.name,
        deployment_name=args.deployment_name,
    )
    print(f"Deployment state: {deployment.provisioning_state}")

    print("Directing 100% of traffic to the deployment...")
    set_traffic_to_deployment(ml_client, endpoint.name, args.deployment_name)

    endpoint = ml_client.online_endpoints.get(name=endpoint.name)
    print(f"Deployment complete. Scoring URI: {endpoint.scoring_uri}")


if __name__ == "__main__":
    main()
