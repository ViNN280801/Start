#!/bin/bash

docker login

docker build -f docker/Dockerfile.fedora.40.base.image -t start-fedora.40-base-image .
docker tag start-fedora.40-base-image vladislavsemykin18/start-fedora.40:omp_v2
docker push vladislavsemykin18/start-fedora.40:omp_v2
docker rmi -f start-fedora.40-base-image

docker build -f docker/Dockerfile.ubuntu.22.04.base.image -t start-ubuntu22.04-base-image .
docker tag start-ubuntu22.04-base-image vladislavsemykin18/start-ubuntu22.04:omp_v2
docker push vladislavsemykin18/start-ubuntu22.04:omp_v2
docker rmi -f start-ubuntu22.04-base-image
