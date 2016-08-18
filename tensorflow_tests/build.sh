#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
docker rm -f tensor
docker rmi -f tensor-gpu
docker build -t tensor-gpu $DIR/dockerized_tensorflow/

