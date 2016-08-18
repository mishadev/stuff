#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
nvidia-docker run -it -v "$DIR/scripts:/scripts" --name tensor tensor-gpu 
