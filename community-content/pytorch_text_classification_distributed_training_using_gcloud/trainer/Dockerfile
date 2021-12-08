# Use pytorch GPU base image
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-8

# set working directory
WORKDIR /app

# Install required packages
RUN echo "deb https://packages.cloud.google.com/apt google-fast-socket-unstable main" | tee /etc/apt/sources.list.d/google-fast-socket.list && \
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
  apt-get update && \
  apt-get remove -y google-fast-socket && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y google-reduction-server

RUN pip install google-cloud-storage transformers datasets tqdm cloudml-hypertune

COPY ./launcher.sh /app/launcher.sh
COPY ./run_glue.py /app/run_glue.py

ENV NCCL_DEBUG=INFO

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["/app/launcher.sh"]
