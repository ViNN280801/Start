#!/bin/bash

# Adding docker to the host X-server to turn on the graphical env
xhost +local:docker

# Build the Docker image
docker build -t startim .

# Rebuild container from scratch each time. (2>/dev/null || true) avoid failure if in we have error in `stderr`
docker stop start_container 2>/dev/null || true
docker rm start_container 2>/dev/null || true

# Run the Docker container with graphical environment support
docker run -it --name start_container -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev/dri:/dev/dri startim

# To remove this image and container execute the following commands
# docker rm start_container 2>/dev/null || true
# docker rmi startim 2>/dev/null || true

# To push image to the dockerhub:
# docker login
# docker tag startim <name_of_the_image_in_dockerhub>
# docker push <name_of_the_image_in_dockerhub>
