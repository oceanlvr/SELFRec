# docker run --rm --gpus all
FROM tensorflow/tensorflow:1.11.0-cuda11.3-cudnn8-devel AS TF
RUN mkdir -p /workspace
WORKDIR /workspace
COPY . /workspace

