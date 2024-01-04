FROM --platform=linux/amd64 python:slim

RUN apt-get update && apt-get -y upgrade \
&& apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    gcc \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY . /src/

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda create -c conda-forge -n dadvi "pymc=5.6" bambi python=3.9 pystan=2.19.1.1 -y && \
    conda activate dadvi && \
    cd /src && \
    pip install -e .[viabel] && \
    git clone https://github.com/jhuggins/viabel.git && \
    cd viabel && \
    git checkout v0.5.1 && \
    pip install -e . && \
    pip install "pandas<2.0.0" && \
    pip install "jax==0.4.14" "jaxlib==0.4.14"

RUN apt-get update && apt-get install -y vim less
