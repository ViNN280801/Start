#!/bin/bash

DOCKERFILE=Dockerfile.ubuntu.executable
IMAGE_NAME=startim_executable
CONTAINER_NAME=start_container_executable

# Adding docker to the host X-server to turn on the graphical env
xhost +local:docker

# Build the Docker image
docker build -f $DOCKERFILE -t $IMAGE_NAME .

# Rebuild container from scratch each time. (2>/dev/null || true) avoid failure if in we have error in `stderr`
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Run the Docker container with graphical environment support
docker run -it --name $CONTAINER_NAME -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev/dri:/dev/dri $IMAGE_NAME

# To remove this image and container execute the following commands
# docker rm $CONTAINER_NAME 2>/dev/null || true
# docker rmi $IMAGE_NAME 2>/dev/null || true

# To push image to the dockerhub:
# docker login
# docker tag $IMAGE_NAME <dockerhub_user>/<image_name>:<tag>
# docker push <dockerhub_user>/<image_name>:<tag>
