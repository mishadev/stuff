#!/bin/bash

docker run -ti --rm -v $(pwd):/home/developer/workspace -v $(pwd)/vimrc:/ext/ jare/vim-bundle
