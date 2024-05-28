# Base container name
ARG BASE_NAME=python:3.11

FROM $BASE_NAME as base

ARG PACKAGE_NAME="triton-exploration"

# Upgrade pytorch version
RUN pip install --upgrade torch torchvision

# Install python packages
WORKDIR /app/${PACKAGE_NAME}
COPY ./requirements.txt /app/${PACKAGE_NAME}/requirements.txt

RUN pip install -r requirements.txt

# Copy all files to the container
COPY scripts /app/${PACKAGE_NAME}/scripts
COPY benchmark /app/${PACKAGE_NAME}/benchmark
# COPY src /app/${PACKAGE_NAME}/src
WORKDIR /app/${PACKAGE_NAME}

# Build the benchmark
# RUN /app/${PACKAGE_NAME}/scripts/compile-rocblas.sh

# Set the entrypoint
RUN chmod a+x /app/${PACKAGE_NAME}/scripts/start.sh

ENV PACKAGE_NAME=$PACKAGE_NAME
ENTRYPOINT ["/app/triton-exploration/scripts/start.sh"]