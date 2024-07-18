FROM pytorch/pytorch:latest

RUN apt-get update \
     && apt-get install -y \
        libgl1-mesa-glx \
        libx11-xcb1 \
        gcc \
        g++ \
        vim \ 
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*

RUN /opt/conda/bin/conda install --yes \
    astropy \
    matplotlib \
    pandas \
    scikit-learn \
    scikit-image 

RUN pip install torch
RUN pip install Ninja
WORKDIR //usr/local/cuda
ADD . //usr/local/cuda

