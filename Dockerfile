# Use TensorFlow GPU image with Jupyter support
FROM tensorflow/tensorflow:2.17.0-gpu-jupyter

# Suppress unnecessary TensorFlow logs
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV TF_CPP_MIN_LOG_LEVEL=3

# Set environment variables for CUDA
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Upgrade pip and install dependencies
WORKDIR /workspace
USER root

COPY requirements.txt .

RUN apt-get update && apt-get install -y graphviz
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Start Jupyter Notebook when container runs
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

