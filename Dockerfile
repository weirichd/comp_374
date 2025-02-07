# Start with the TensorFlow GPU Jupyter image

FROM tensorflow/tensorflow:latest-gpu-jupyter



# Set environment variables for CUDA
ENV CUDA_VERSION=12.3
ENV CUDNN_VERSION=9
ENV PATH="/usr/local/cuda-${CUDA_VERSION}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/lib64:${LD_LIBRARY_PATH}"

# Upgrade pip and install dependencies
WORKDIR /workspace
USER root

COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Verify installation
RUN python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
