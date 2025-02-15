# Name of the Docker image and container
IMAGE_NAME = my-tensorflow-gpu-jupyter
CONTAINER_NAME = tensorflow-gpu-jupyter-container
TF_VERSION = 2.17.0  # Change this to update TensorFlow

# Default target: Build and check GPU availability
.PHONY: all 
all: build check-gpu

# Build the Docker image
.PHONY: build
build: Dockerfile requirements.txt
	@echo "Building Docker Image: $(IMAGE_NAME)..."
	docker build --build-arg TF_VERSION=$(TF_VERSION) -t $(IMAGE_NAME) .

# Run the container interactively with GPU support
.PHONY: run 
run: check-gpu
	@echo "Running Docker Container with GPU..."
	docker run --gpus all -it --rm --name $(CONTAINER_NAME) -p 8888:8888 \
		-v $(shell pwd):/workspace $(IMAGE_NAME)

# Run arbitrary commands inside the container
.PHONY: run-cmd
run-cmd: check-gpu
	@echo "Running command inside container: $(filter-out $@,$(MAKECMDGOALS))..."
	docker run --gpus all -it --rm --name $(CONTAINER_NAME) \
		-v $(shell pwd):/workspace $(IMAGE_NAME) $(filter-out $@,$(MAKECMDGOALS))

# Rebuild the image (force update)
.PHONY: rebuild
rebuild:
	@echo "Rebuilding Docker Image from scratch..."
	docker build --no-cache --build-arg TF_VERSION=$(TF_VERSION) -t $(IMAGE_NAME) .

# Stop and remove any running container with the same name
.PHONY: clean
clean:
	@echo "Stopping and cleaning up container..."
	docker rm -f $(CONTAINER_NAME) || true
	docker rmi -f $(IMAGE_NAME) || true

# Show running containers
.PHONY: ps
ps:
	docker ps -a | grep $(CONTAINER_NAME) || echo "No running container"

# Check if GPUs are visible
.PHONY: check-gpu
check-gpu:
	@echo "Checking if GPUs are available..."
	@if ! docker run --gpus all --rm $(IMAGE_NAME) python3 -c "import tensorflow as tf; assert tf.config.list_physical_devices('GPU'), '\033[0;31mERROR: No GPUs detected inside container!\033[0m'" 2>/dev/null; then \
		echo "\033[0;31mERROR: NO GPUs DETECTED INSIDE CONTAINER!\033[0m"; \
		exit 1; \
	fi
	@echo "GPUs are available."

# Easily update TensorFlow version and rebuild
.PHONY: update-tensorflow
update-tensorflow:
	@echo "Updating TensorFlow to version $(TF_VERSION)..."
	docker build --no-cache --build-arg TF_VERSION=$(TF_VERSION) -t $(IMAGE_NAME) .

# Allow passing additional arguments to run-cmd
%:
	@:

