# H100-optimized base image for InfiniteTalk
# Uses CUDA 12.8.1, PyTorch 2.7 with cu128, and SageAttention compiled with SM90 support

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 as runtime

# Remove any third-party apt sources
RUN rm -f /etc/apt/sources.list.d/*.list

# Set shell and environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV SHELL=/bin/bash
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV TORCH_CUDA_ARCH_LIST="9.0"

# Set working directory
WORKDIR /

# Update and install system packages
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends git wget curl bash libgl1 software-properties-common openssh-server nginx rsync ffmpeg && \
    apt-get install --yes --no-install-recommends build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev git-lfs && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install python3.10-dev python3.10-venv -y --no-install-recommends && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Setup Python 3.10
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py

RUN pip install -U wheel setuptools packaging

# Install PyTorch 2.7 with CUDA 12.8 support
RUN pip install torch==2.7.0 torchvision==0.20.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Install xFormers compatible with PyTorch 2.7 and CUDA 12.8
RUN pip install xformers --index-url https://download.pytorch.org/whl/cu128

WORKDIR /
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git

WORKDIR /InfiniteTalk

RUN pip install misaki[en]
RUN pip install ninja psutil packaging

# Install Flash Attention 2
RUN pip install flash_attn==2.7.4.post1 --no-build-isolation

# Install Triton 3.3.0 (required for SageAttention on H100)
RUN pip install triton==3.3.0

# Clone and build SageAttention with SM90 support for H100
WORKDIR /tmp
RUN git clone https://github.com/thu-ml/SageAttention.git && \
    cd SageAttention && \
    # Modify setup.py to enable SM90
    sed -i 's/# HAS_SM90 = False/HAS_SM90 = True/g' setup.py && \
    sed -i 's/# SUPPORTED_ARCHS = {"8.0", "8.6", "8.9", "9.0"}/SUPPORTED_ARCHS = {"9.0"}/g' setup.py && \
    # Build with SM90 support
    TORCH_CUDA_ARCH_LIST="9.0" python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    cd / && rm -rf /tmp/SageAttention

WORKDIR /InfiniteTalk

# Install other requirements
RUN pip install -r requirements.txt
RUN pip install librosa ffmpeg
RUN pip uninstall -y transformers
RUN pip install transformers==4.48.2
RUN pip install runpod websocket-client
RUN pip install -U "huggingface_hub[hf_transfer]"

# Verify H100 compatibility
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'PyTorch version: {torch.__version__}')"

# Build instructions:
# docker build -t <your-dockerhub-username>/infinitetalk-h100-base:1.0 -f base_h100.Dockerfile .
# docker push <your-dockerhub-username>/infinitetalk-h100-base:1.0
