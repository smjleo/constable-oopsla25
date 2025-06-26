FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# create a non-root user
RUN groupadd -r myuser && \
    useradd -r -g myuser -d /home/myuser -m myuser

# install all apt dep
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      apt-utils tzdata autoconf cmake gcc g++ gfortran \
      ninja-build libopenmpi-dev git wget curl numactl time libomp-dev && \
    rm -rf /var/lib/apt/lists/*

USER myuser
WORKDIR /home/myuser

ENV HOME=/home/myuser

# put user local bin on PATH
RUN mkdir -p .local/bin
ENV PATH="$HOME/.local/bin:$PATH"

# Install Bazelisk
RUN echo "Installing Bazelisk…" && \
    curl -fLO https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64 && \
    mv bazelisk-linux-amd64 .local/bin/bazel && \
    chmod +x .local/bin/bazel && \
    mkdir -p .baztmp

 # Install Rust + cxxbridge
ENV RUSTUP_HOME=$HOME/.local/.rustup \
    CARGO_HOME=$HOME/.local/.cargo

# create the dirs up front under $HOME
RUN mkdir -p $CARGO_HOME/bin $RUSTUP_HOME && \
    echo "Installing Rust…" && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="$CARGO_HOME/bin:$PATH"
RUN rustup default stable && \
    cargo install cxxbridge-cmd

# Build Enzyme-JAX + Constable
RUN echo "Building Enzyme-JAX + Constable…" && \
    git clone https://github.com/EnzymeAD/Enzyme-JAX && \
    cd Enzyme-JAX && \
    git checkout 9322d4f && \
    touch src/enzyme_ad/jax/deps/tensat/cargo-bazel-lock.json

RUN mkdir -p /home/myuser/.eqsat-tmp /home/myuser/Enzyme-JAX/.eqsat-tmp
ENV TMPDIR=/home/myuser/.eqsat-tmp \
    TMP=/home/myuser/.eqsat-tmp \
    TEMP=/home/myuser/.eqsat-tmp \
    CARGO_BAZEL_REPIN=true

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      clang build-essential python3-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*
USER myuser 

WORKDIR /home/myuser/Enzyme-JAX
ENV CONDA_DIR=/home/myuser/miniconda3
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
     -o miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh

SHELL ["/bin/bash", "-c"]
RUN source "$CONDA_DIR/etc/profile.d/conda.sh" && \
    conda create -y -n enzyme python=3.11 pip && \
    conda activate enzyme && \
    conda clean -afy && \
    conda activate enzyme

WORKDIR /home/myuser/Enzyme-JAX
RUN source "$CONDA_DIR/etc/profile.d/conda.sh" && \
    conda activate enzyme && \
    ./build.sh

RUN source "$CONDA_DIR/etc/profile.d/conda.sh" && \
    conda activate enzyme && \
    python --version && \
    pip --version && \
    pip install absl-py flax transformers pillow ortools jax==0.4.31 "jax[cuda]==0.4.31" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    conda install -c conda-forge libstdcxx-ng

ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Inject conda hook + auto-activate into ~/.bashrc
RUN echo "" >> $HOME/.bashrc && \
    echo "# >>> conda initialize >>>" >> $HOME/.bashrc && \
    echo ". $CONDA_DIR/etc/profile.d/conda.sh && conda activate enzyme" >> $HOME/.bashrc && \
    echo "# <<< conda initialize <<<" >> $HOME/.bashrc

WORKDIR /home/myuser
RUN source "$CONDA_DIR/etc/profile.d/conda.sh" && \
    conda activate enzyme && \
    pip install git+https://github.com/wsmoses/jax-md@45059b8f63dad0b5cb171feafff71b82162487e7 && \
    pip install jraph && \
    git clone https://github.com/wsmoses/maxtext && \
    pip install -e maxtext && pip install jax==0.4.31 "jax[cuda]==0.4.31" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip install aqtp tensorboardx datasets

CMD ["echo", "Constable OOPSLA 2025 Docker Image"]
WORKDIR /home/myuser/Enzyme-JAX
RUN ls experiment/
