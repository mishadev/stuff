#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
docker build -t tensor-gpu $DIR/dockerized_tensorflow/
