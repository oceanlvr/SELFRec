# docker build -t oceanlvr/selfrectorch .
# docker run -e WANDB_DOCKER=oceanlvr/selfrectorch -itd --gpus all --name oyx_selfrec oceanlvr/selfrectorch

FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
RUN mkdir -p /workspace
WORKDIR /workspace
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . .
