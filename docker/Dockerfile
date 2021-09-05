FROM nvidia/cuda:11.4.0-devel-ubuntu20.04

# Force color prompt in terminal.
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc

# install utility
RUN apt-get update && \
    apt-get install -y \
    wget \
    curl \
    git \
    vim \
    python3-pip

# install python package
WORKDIR /root/requirements
ADD requirements.txt /root/requirements
RUN pip3 install -r requirements.txt

# install fish
RUN apt-get update && apt-get install -y fish

WORKDIR /root/workspace
