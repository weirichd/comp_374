# Name of the Docker image
IMAGE_NAME = my-tensorflow-gpu-jupyter
CONTAINER_NAME = tensorflow-gpu-jupyter-container

# Default target: Build the Docker image
.PHONY: all
all: build

# Build the Docker image (checks if requirements.txt changed)
.PHONY: build
build: Dockerfile requirements.txt
	docker build -t $(IMAGE_NAME) .

# Run the container interactively with GPU support and expose Jupyter
.PHONY: run
run:
	docker run --gpus all -it --rm --name $(CONTAINER_NAME) -p 8888:8888 -v $(shell pwd):/tf/notebooks $(IMAGE_NAME)

# Rebuild the image (force update)
.PHONY: rebuild
rebuild:
	docker build --no-cache -t $(IMAGE_NAME) .

# Stop and remove any running container with the same name
.PHONY: clean
clean:
	docker rm -f $(CONTAINER_NAME) || true
	docker rmi -f $(IMAGE_NAME) || true

# Show running containers
.PHONY: ps
ps:
	docker ps -a | grep $(CONTAINER_NAME) || echo "No running container"

