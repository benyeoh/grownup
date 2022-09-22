FROM dragnet:latest

USER root
RUN apt-get update && apt-get install -y --no-install-recommends libmagic1

USER dragnet

RUN pip install python-magic==0.4.18

RUN mkdir /tmp/dragnet && mv * /tmp/dragnet/ && mv /tmp/dragnet .
COPY ./*.py $HOME/
ENV PATH="$HOME/dragnet/py/bin:$HOME/dragnet/miniconda3/bin:$PATH"
