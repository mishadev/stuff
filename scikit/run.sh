#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
docker run -it -v "$DIR/scripts:/scripts" --name sklearn scikit-learn
