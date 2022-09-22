FROM nvidia/cuda:11.2.1-devel-ubuntu18.04

# Replace with owner UID/GID
ARG UID=1000
ARG GID=1000

# add new sudo user
ENV USERNAME klass
ENV HOME /home/$USERNAME
ENV DEBIAN_FRONTEND=noninteractive

RUN useradd -l -m $USERNAME && \
    echo "$USERNAME:$USERNAME" | chpasswd && \
    usermod --shell /bin/bash $USERNAME && \
    usermod -aG sudo $USERNAME && \
    mkdir /etc/sudoers.d && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    usermod  --uid $UID $USERNAME && \
    groupmod --gid $GID $USERNAME

# Essentials
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    apt-utils \
    git \
    bash-completion \
    libgtk-3-0 \
    locales \
    wget \
    gzip \
    tk \
    vim \
    python \
    python3.7 \
    python3.7-dev \
    python3.7-tk \
    python3.7-venv \
    apt-transport-https \
    ca-certificates \
    curl \
    libmagic1

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py

# Set locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

USER klass

curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
RUN sudo -H pip install readability-lxml==0.8.1 python-magic==0.4.18 beautifulsoup4==4.9.2 readabilipy-0.2.0

RUN mkdir -p $HOME/readability
COPY --chown=klass ./ $HOME/readability

WORKDIR $HOME/readability
