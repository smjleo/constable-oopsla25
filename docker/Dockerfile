FROM ubuntu:noble
SHELL ["/bin/bash", "-c"]

CMD ["echo", "Constable OOPSLA 2025 Docker Image"]

RUN groupadd -r myuser && useradd -r -g myuser myuser

RUN apt-get update
RUN apt-get -y install apt-utils
RUN apt-get -y install tzdata --assume-yes
RUN apt-get -y install autoconf cmake gcc g++ gfortran ninja-build libopenmpi-dev
RUN apt-get -y install git 
RUN apt-get -y install wget
RUN apt-get -y install numactl
RUN apt-get -y install python3
RUN apt-get -y install time
RUN apt-get -y install libomp-dev

RUN mkdir /home/myuser
RUN chown myuser /home/myuser

USER myuser

WORKDIR /home/myuser
RUN mkdir -p .local/bin
ENV PATH "$PATH:/home/myuser/.local/bin"

# Install Bazel
CMD ["echo", "Installing Bazel"]
RUN curl -fLO https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64
RUN mv bazel* .local/bin/bazel
RUN chmod +x .local/bin/bazel
RUN mkdir -p .baztmp

# Install Rust, Cargo and cxxbridge
CMD ["echo", "Installing Rust"]
ENV RUSTUP_HOME /home/myuser/.local/.rustup
ENV CARGO_HOME /home/myuser/.local/.cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH "$PATH:/home/myuser/.local/.cargo/bin"
RUN rustup default stable
RUN cargo install cxxbridge-cmd

# Build Enzyme-JAX + Constable
CMD ["echo", "Building Constable"]
WORKDIR /home/myuser
RUN git clone https://github.com/EnzymeAD/Enzyme-JAX

WORKDIR /home/myuser/Enzyme-JAX
RUN git checkout 482e2c4
RUN touch src/enzyme_ad/jax/deps/tensat/cargo-bazel-lock.json
# TODO: env vars / build flags?
RUN mkdir .eqsat-tmp
ENV TMPDIR /home/myuser/Enzyme-JAX/.eqsat-tmp
ENV TMP home/myuser/Enzyme-JAX/.eqsat-tmp
ENV TEMP /home/myuser/Enzyme-JAX/.eqsat-tmp

ENV CARGO_BAZEL_REPIN true
ENV HERMETIC_PYTHON_VERSION 3.11
RUN bazel build :wheel --check_visibility=false
RUN pip3 install ortools
RUN pip3 install bazel-bin/enzyme_ad-0.0.8-py311-none-macosx_11_0_arm64.whl
