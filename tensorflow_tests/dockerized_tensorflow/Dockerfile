FROM tensorflow/tensorflow:latest-gpu

RUN dpkg --add-architecture i386
RUN apt-get update
RUN sudo apt-get -y install     \
    python2.7-dev               \
    build-essential             \
    python-cffi                 \
    libffi-dev                  \
    libssl-dev                  \
    pkg-config

RUN pip install --upgrade pip
RUN pip --no-cache-dir install  \
    nltk                        \
    'requests[security]'        \
    tweepy
