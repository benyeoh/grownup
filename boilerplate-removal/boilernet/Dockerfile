FROM tensorflow/tensorflow:2.1.0-gpu

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
        gnupg \
        graphviz \
        curl

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py

USER klass

# Do install first as we want to cache this
COPY ./boilernet/requirements.txt /tmp/requirements.txt

RUN sudo -H pip install -r /tmp/requirements.txt
RUN sudo -H pip install h5py==2.10.0

# Finally copy entire repo
COPY ./ $HOME/boilernet

WORKDIR $HOME/
