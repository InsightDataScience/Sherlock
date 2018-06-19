FROM continuumio/miniconda:4.4.10

# utils
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

RUN conda install gxx_linux-64

RUN apt-get install -y --force-yes default-libmysqlclient-dev mysql-client build-essential

# Grab requirements.txt.
COPY requirements.txt /tmp/requirements.txt

# Install dependencies
RUN pip install -qr /tmp/requirements.txt

# create a user for web server
RUN adduser --disabled-password --gecos "" foo

COPY ./ /opt/src

WORKDIR /opt/src