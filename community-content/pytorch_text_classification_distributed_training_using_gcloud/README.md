# PyTorch on Google Cloud: Hugging Face Transformers at Scale

## Overview

The directory provides code to fine tune a transformer model ([BERT-large](https://huggingface.co/bert-large-uncased)) from Huggingface Transformers Library for GLUE tasks.  [BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html) (Bidirectional Encoder Representations from Transformers) is a transformers model pre-trained on a large corpus of unlabeled text in a self-supervised fashion. [GLUE](https://gluebenchmark.com/) benchmark is made up of a total of 9 different tasks. In this sample, we use the MNLI task. We show you launching a distributed training job on Vertex AI based on the trainer script [run_glue.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py) from Huggingface Transformers Library using [Vertex Training custom containers](https://cloud.google.com/vertex-ai/docs/training/create-custom-container?hl=hr). 

## Prerequisites

* Setup your project by following the instructions from [documentation](https://cloud.google.com/vertex-ai/docs/start/cloud-environment)
* [Setup docker with Cloud Container Registry](https://cloud.google.com/container-registry/docs/pushing-and-pulling)
* Change the directory to this sample and run

`Note:` These instructions are used for local testing. When you submit a training job, no code will be executed on your local machine.
  

## Directory Structure

* `trainer` directory: training scripts to be packaged as a custom container.
* `scripts` directory: command-line scripts to train the model on Vertex AI.

### Trainer
| File Name | Purpose |
| :-------- | :------ |
| [Dockerfile](trainer/Dockerfile) | Dockerfile of the custom container. |
| [launcher.sh](trainer/launcher.sh) | Entrypoint shell script of the custom container. |
| [run_glue.py](trainer/run_glue.py) | Trainer script from [Huggingface Transformers Library](https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py). |

### Scripts

* [train-cloud.sh](scripts/train-cloud.sh) This script builds your Docker image locally, pushes the image to Container Registry and submits a custom container training job to Vertex AI.

Please read the [documentation](https://cloud.google.com/vertex-ai/docs/training/containers-overview?hl=hr) on Vertex Training with Custom Containers for more details.

## How to run

Once the prerequisites are satisfied, you may:

1. For local testing, run:
    ```
    CUSTOM_TRAIN_IMAGE_URI='gcr.io/{PROJECT_ID}/pytorch_gpu_train_{APP_NAME}'
    cd ./trainer/ && docker build -f Dockerfile -t $CUSTOM_TRAIN_IMAGE_URI .
    docker run --gpus all -it --rm $CUSTOM_TRAIN_IMAGE_URI
    ```
2. For cloud testing, run:
    ```
    source ./scripts/train-cloud.sh
    ```

## Run on multiple workers
The provided trainer code runs on GPUs and scales to multiple workers if the worker pools are specified. It also leverages [Reduction Server](https://cloud.google.com/blog/products/ai-machine-learning/faster-distributed-training-with-google-clouds-reduction-server) to speed up the distributed GPU training.

To run the trainer code on a different GPU configuration, make the following changes to the [train-cloud.sh](scripts/train-cloud.sh) script.
* Update the second [`workerPoolSpecs`](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/CustomJobSpec#workerpoolspec) to change the type and number of GPUs.
* Update the third [`workerPoolSpecs`](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/CustomJobSpec#workerpoolspec) to change the number of [Reduction Server](https://cloud.google.com/vertex-ai/docs/training/distributed-training#reduce_training_time_with_reduction_server) instances.

Then, run the script to submit a Custom Job on Vertex Training job:
```
source ./scripts/train-cloud.sh
```

### Versions
This script uses the Deep Learning Containers for PyTorch 1.9.
* `gcr.io/deeplearning-platform-release/pytorch-gpu.1-9`
