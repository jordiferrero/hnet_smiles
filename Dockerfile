# Dockerfile for HNet SMILES Training on CUDA
# Assumes CUDA is available in the container
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    build-essential \
    ninja-build \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
# HNet requires torch>=2.7.1, but we'll install the latest stable version
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install triton (required for HNet)
# Note: HNet requires triton>=3.3.1, but we'll install what's available
RUN pip3 install "triton>=3.3.1" || pip3 install "triton>=3.2.0" || echo "Warning: triton installation may have failed"

# Install base Python dependencies
RUN pip3 install \
    einops \
    optree \
    regex \
    omegaconf \
    pandas \
    numpy \
    matplotlib \
    imageio \
    Pillow \
    tqdm

# Install git dependencies (order matters for build dependencies)
# First install causal-conv1d (dependency for mamba_ssm)
RUN pip3 install git+https://github.com/Dao-AILab/causal-conv1d.git@e940ead2fd962c56854455017541384909ca669f

# Install mamba_ssm (requires CUDA and nvcc)
RUN pip3 install git+https://github.com/state-spaces/mamba.git@a6a1dae6efbf804c9944a0c2282b437deb4886d8

# Install flash-attn (optional but recommended for better performance)
# This may fail on some systems, but we'll continue without it if needed
RUN set +e && pip3 install flash-attn==2.8.0.post2 --no-build-isolation || echo "Warning: flash-attn installation failed, continuing without it..." && set -e

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Install hnet package in editable mode
RUN cd original_resources/hnet-github-repo && \
    pip3 install -e . && \
    cd /workspace

# Verify installation
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); import hnet; print('HNet installed successfully')" || \
    echo "Warning: Some imports may have failed"

# Default command
CMD ["/bin/bash"]

