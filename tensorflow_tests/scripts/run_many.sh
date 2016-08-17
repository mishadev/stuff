#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
for i in {1..2}
do
     python $DIR/main.py
done

