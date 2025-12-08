# Use Ubuntu as base image
FROM ubuntu:latest

# Set the maintainer
LABEL maintainer="Your Name wang.tw.ted@gmail.com"

# Install Development Tools and dependencies
RUN apt update && apt install -y \
    build-essential cmake automake autoconf \
    sudo \
    gcc \
    g++ \
    make \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-numpy \
    python3-pandas \
    python3-opencv \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libeigen3-dev \
    # vim \
    # git \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Set up working directory
WORKDIR /workspace

ENV SHELL=/bin/bash
ENV TERM=xterm-256color

# USER $USERNAME

# COPY .bashrc /home/$USERNAME/.bashrc.conf
# RUN cat /home/$USERNAME/.bashrc.conf >> /home/$USERNAME/.bashrc

# Set the default command to bash
CMD ["/bin/bash", "-ic"]