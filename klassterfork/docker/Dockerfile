FROM nvidia/cuda:11.2.1-devel-ubuntu18.04

# Replace with owner UID/GID
ARG UID=1000
ARG GID=1000

# add new sudo user
ENV USERNAME forever.young
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
        curl \
        libmagic1 \
        protobuf-compiler

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1; \
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py "pip < 20.3"; pip --version;

RUN apt-get update && apt-get install -y --no-install-recommends \
        libcudnn8=8.1.1.33-1+cuda11.2 \
        libcudnn8-dev=8.1.1.33-1+cuda11.2

#RUN apt-get install -y --no-install-recommends libmagic1

# Set locale
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

USER forever.young
ENV PATH="$PATH:$HOME/.local/bin"

# Do install first as we want to cache this
RUN mkdir -p $HOME/klassterfork/
COPY ./requirements.txt $HOME/klassterfork/

RUN pip install -r $HOME/klassterfork/requirements.txt

COPY ./requirements_ext.txt $HOME/klassterfork/
RUN pip install -r $HOME/klassterfork/requirements_ext.txt

# Finally copy entire repo
# Separate into sections so that we can cache the large stuff earlier for faster building and pushing of image
#COPY ./python/tests/data $HOME/klassterfork/python/tests/data
COPY ./docker $HOME/klassterfork/docker
COPY ./*.* $HOME/klassterfork/
COPY ./scripts $HOME/klassterfork/scripts
COPY ./ktf $HOME/klassterfork/ktf
#COPY ./python/tests/*.* $HOME/klassterfork/python/tests/
COPY ./runs $HOME/klassterfork/runs

WORKDIR $HOME/klassterfork
