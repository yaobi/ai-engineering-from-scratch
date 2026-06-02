# Docker Helper

## 1. Image vs Container

Image:

A template used to create containers.

Container:

A running instance created from an image.

Example:

hello-world is an image.
docker run hello-world creates and runs a container.

---

## 2. Dockerfile

A Dockerfile describes how to build an image.

Important instructions:

FROM: choose a base image.
RUN: execute commands while building the image.
WORKDIR: set the working directory.
CMD: default command when the container starts.

---

## 3. Volume mount

A volume mount maps a host directory into a container.

Example:

-v $(pwd):/workspace

This maps the current project directory into /workspace inside the container.

Use volume mounts for code, datasets, and model files so they are not lost when the container stops.

---

## 4. CPU vs GPU Docker

This machine does not have an NVIDIA GPU.

Use CPU images such as:

python:3.12-slim

Skip CUDA images and --gpus all unless an NVIDIA GPU is available.

---

## 5. Common Docker commands

docker --version
docker run hello-world
docker images
docker ps -a
docker container prune
docker build -t ai-dev-cpu -f phases/00-setup-and-tooling/07-docker-for-ai/code/Dockerfile.cpu .
docker run --rm ai-dev-cpu python -c "import numpy as np; print(np.__version__)"

---

## 6. Network and proxy issues

If Docker cannot pull images from Docker Hub, configure Docker Desktop proxy settings.

Common proxy:

http://127.0.0.1:7890

Then retry:

docker pull python:3.12-slim
