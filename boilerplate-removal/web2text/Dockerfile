FROM tensorflow/tensorflow:1.15.0-gpu

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

RUN echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" | sudo tee /etc/apt/sources.list.d/sbt.list
RUN echo "deb https://repo.scala-sbt.org/scalasbt/debian /" | sudo tee /etc/apt/sources.list.d/sbt_old.list
RUN curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | sudo apt-key add
RUN apt-get update && apt-get install -y openjdk-8-jdk sbt=1.3.3

USER klass

# Do install first as we want to cache this
#COPY ./boilernet/requirements.txt /tmp/requirements.txt

#RUN sudo -H pip install -r /tmp/requirements.txt
#RUN sudo -H pip install h5py==2.10.0
RUN sudo -H pip install future==0.18.2 numpy==1.18.0 click==8.0.1 tensorflow==1.15.0 tensorflow-gpu==1.15.0

# Finally web2text submodule
RUN mkdir -p $HOME/web2text/web2text
COPY --chown=klass:klass ./web2text/ $HOME/web2text/web2text/

# Initialize sbt
RUN cd $HOME/web2text/web2text && sbt -v 'compile'

COPY --chown=klass:klass ./*.* $HOME/web2text/
COPY --chown=klass:klass ./scala $HOME/web2text/scala
RUN cd $HOME/web2text/web2text/src/main/scala && ln -s ../../../../scala/bench

WORKDIR $HOME/

