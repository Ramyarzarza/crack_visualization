"""
Run this script ONCE to deploy a crack-detection model to a SageMaker
Serverless Inference endpoint.

Usage:
    cd sagemaker/
    python deploy.py \
        --role   arn:aws:iam::123456789012:role/SageMakerCrackDetectionRole \
        --bucket your-s3-bucket-name \
        --model  "Generalized_dataset_bestmodel_real+simple background_more epochs.pt" \
        --region us-east-1
"""
import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import boto3

# ── Pre-built SageMaker PyTorch CPU container (PyTorch 2.1, Python 3.10) ──
PYTORCH_IMAGE = (
    "763104351884.dkr.ecr.{region}.amazonaws.com"
    "/pytorch-inference:2.1.0-cpu-py310"
)

MODEL_NAME     = "crack-detection-model"
CONFIG_NAME    = "crack-detection-serverless-config"
ENDPOINT_NAME  = "crack-detection-endpoint"
S3_KEY         = "crack-detection/model.tar.gz"


def build_tarball(model_path: Path, inference_py: Path, out: Path) -> None:
    """Package model.pt + inference.py into model.tar.gz at root level."""
    with tarfile.open(out, "w:gz") as tar:
        tar.add(model_path,    arcname="model.pt")
        tar.add(inference_py,  arcname="inference.py")
    print(f"  Packaged: {out} ({out.stat().st_size / 1e6:.1f} MB)")


def upload_to_s3(local: Path, bucket: str, key: str, region: str) -> str:
    s3 = boto3.client("s3", region_name=region)
    # Create bucket if it doesn't exist (only works in same region)
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(Bucket=bucket,
                             CreateBucketConfiguration={"LocationConstraint": region})
        print(f"  Created bucket: {bucket}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass
    except Exception:
        pass  # bucket already exists and owned by you elsewhere

    print(f"  Uploading to s3://{bucket}/{key} ...")
    s3.upload_file(str(local), bucket, key)
    return f"s3://{bucket}/{key}"


def deploy(role: str, bucket: str, model_file: str, region: str) -> None:
    base = Path(__file__).parent.parent
    model_path    = base / "Models" / model_file
    inference_py  = Path(__file__).parent / "inference.py"

    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")
    if not inference_py.exists():
        sys.exit(f"inference.py not found: {inference_py}")

    # 1. Build tarball
    print("\n[1/4] Building model.tar.gz ...")
    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / "model.tar.gz"
        build_tarball(model_path, inference_py, tarball)

        # 2. Upload to S3
        print("\n[2/4] Uploading to S3 ...")
        s3_uri = upload_to_s3(tarball, bucket, S3_KEY, region)

    sm = boto3.client("sagemaker", region_name=region)
    image = PYTORCH_IMAGE.format(region=region)

    # 3. Create SageMaker Model
    print(f"\n[3/4] Creating SageMaker Model '{MODEL_NAME}' ...")
    try:
        sm.delete_model(ModelName=MODEL_NAME)
        print("  (deleted old model)")
    except sm.exceptions.ClientError:
        pass

    sm.create_model(
        ModelName=MODEL_NAME,
        PrimaryContainer={
            "Image": image,
            "ModelDataUrl": s3_uri,
            "Environment": {"SAGEMAKER_PROGRAM": "inference.py"},
        },
        ExecutionRoleArn=role,
    )

    # 4. Endpoint config + deploy
    print(f"\n[4/4] Creating serverless endpoint '{ENDPOINT_NAME}' ...")
    try:
        sm.delete_endpoint_config(EndpointConfigName=CONFIG_NAME)
    except sm.exceptions.ClientError:
        pass

    sm.create_endpoint_config(
        EndpointConfigName=CONFIG_NAME,
        ProductionVariants=[{
            "VariantName":    "AllTraffic",
            "ModelName":      MODEL_NAME,
            "ServerlessConfig": {
                "MemorySizeInMB": 3072,
                "MaxConcurrency": 5,
            },
        }],
    )

    try:
        sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print("  (deleted old endpoint, waiting ...)")
        waiter = sm.get_waiter("endpoint_deleted")
        waiter.wait(EndpointName=ENDPOINT_NAME)
    except sm.exceptions.ClientError:
        pass

    sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=CONFIG_NAME)

    print("\n✅ Deployment started!")
    print("   Check status: AWS Console → SageMaker → Endpoints")
    print(f"   Endpoint name: {ENDPOINT_NAME}")
    print("\n   Once status is 'InService', set this env var and restart the backend:")
    print(f"   export SAGEMAKER_ENDPOINT={ENDPOINT_NAME}")
    print(f"   export AWS_DEFAULT_REGION={region}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role",   required=True, help="SageMaker IAM Role ARN")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--model",  required=True, help="Model filename inside Models/")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()
    deploy(args.role, args.bucket, args.model, args.region)
