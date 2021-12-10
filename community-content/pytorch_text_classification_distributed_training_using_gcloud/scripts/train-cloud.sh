#!/bin/bash
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This script performs cloud training for a PyTorch model.

echo "Submitting PyTorch model training job to Vertex AI"

# PROJECT_ID: Change to your project id
PROJECT_ID=$(gcloud config list --format 'value(core.project)')

# BUCKET_NAME: Change to your bucket name.
BUCKET_NAME="changlan" # <-- CHANGE TO YOUR BUCKET NAME

# JOB_NAME: the name of your job running on AI Platform.
JOB_PREFIX="finetuned-bert-classifier-pytorch-cstm-cntr"
JOB_NAME=${JOB_PREFIX}-$(date +%Y%m%d%H%M%S)-custom-job

# This can be a GCS location to a zipped and uploaded package
PACKAGE_PATH=./trainer

# REGION: select a region from https://cloud.google.com/vertex-ai/docs/general/locations#available_regions
# or use the default '`us-central1`'. The region is where the job will be run.
REGION="us-central1"

# JOB_DIR: Where to store prepared package.
JOB_DIR=/tmp/${BUCKET_NAME}/${JOB_NAME}

# IMAGE_REPO_NAME: set a local repo name to distinquish our image
IMAGE_REPO_NAME=pytorch_gpu_train_finetuned-bert-classifier

# IMAGE_TAG: an easily identifiable tag for your docker image
IMAGE_TAG=latest

# IMAGE_URI: the complete URI location for Cloud Container Registry
CUSTOM_TRAIN_IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_TAG}

# Build the docker image
docker build --no-cache -f ./trainer/Dockerfile -t $CUSTOM_TRAIN_IMAGE_URI ./trainer

# Deploy the docker image to Cloud Container Registry
docker push ${CUSTOM_TRAIN_IMAGE_URI}

cat > config.yaml <<EOF
workerPoolSpecs:
  - machineSpec:
      machineType: a2-highgpu-8g
      acceleratorCount: 8
      acceleratorType: NVIDIA_TESLA_A100
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 500
    replicaCount: 1
    containerSpec:
      imageUri: ${CUSTOM_TRAIN_IMAGE_URI}
      args:
        - --output_dir
        - ${JOB_DIR}
  - machineSpec:
      machineType: a2-highgpu-8g
      acceleratorCount: 8
      acceleratorType: NVIDIA_TESLA_A100
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 500
    replicaCount: 7
    containerSpec:
      imageUri: ${CUSTOM_TRAIN_IMAGE_URI}
      args:
        - --output_dir
        - ${JOB_DIR}
  - machineSpec:
      machineType: n1-highcpu-16
    replicaCount: 24
    containerSpec:
      imageUri: us-docker.pkg.dev/vertex-ai-restricted/training/reductionserver:latest
EOF

# Submit Custom Job to Vertex AI
gcloud beta ai custom-jobs create \
    --display-name=${JOB_NAME} \
    --region ${REGION} \
    --config=config.yaml

rm config.yaml

echo "After the job is completed successfully, model files will be saved at $JOB_DIR/"

# uncomment following lines to monitor the job progress by streaming logs

# Stream the logs from the job
# gcloud ai custom-jobs stream-logs $(gcloud ai custom-jobs list --region=$REGION --filter="displayName:"$JOB_NAME --format="get(name)")

# # Verify the model was exported
# echo "Verify the model was exported:"
# gsutil ls ${JOB_DIR}/