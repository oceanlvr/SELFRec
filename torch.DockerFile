# docker build -t selfrectorch -f torch.DockerFile .
# docker run -itd --gpus all -v /home/oyx/gcl_rc/SELFRec:/workspace --name selfrec selfrectorch

FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime AS TORCH
RUN mkdir -p /workspace
WORKDIR /workspace
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . /workspace


