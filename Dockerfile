# This will be the Dockerfile for event response 

# Start from the jupyter/scipy-notebook image, which includes....
# python3, scipy, pandas, matplotlib, ipywidgets (activated), numba, hdf5, etc...
# For more information... https://hub.docker.com/r/jupyter/scipy-notebook/dockerfile

FROM jupyter/scipy-notebook
WORKDIR /home/jovyan/gnssrefl_jupyter

MAINTAINER = UNAVCO Inc.

USER root

RUN apt-get update && \
    apt-get install -y gcc && \
    apt-get install -y gfortran && \
    rm -rf /var/lib/apt/lists/*

# COPY the bin folder (code that should not be edited by the user) into the docker
COPY --chown=$NB_UID:users /bin /home/jovyan/gnssrefl_jupyter/bin
RUN chmod +x /home/jovyan/gnssrefl_jupyter/bin/*
ENV PYTHONPATH "${PYTHONPATH}:/home/jovyan/gnssrefl_jupyter/bin"

# Copy the data directory
COPY --chown=$NB_UID:users /data /home/jovyan/gnssrefl_jupyter/data
RUN chmod +x /home/jovyan/gnssrefl_jupyter/data/*
ENV PYTHONPATH "${PYTHONPATH}:/home/jovyan/gnssrefl_jupyter/data"

# Copy the orbits directory
COPY --chown=$NB_UID:users /orbits /home/jovyan/gnssrefl_jupyter/orbits
ENV PYTHONPATH "${PYTHONPATH}:/home/jovyan/gnssrefl_jupyter/orbits"

ENV ORBITS=/home/jovyan/gnssrefl_jupyter/orbits
ENV REFL_CODE=/home/jovyan/gnssrefl_jupyter
ENV EXE=/home/jovyan/gnssrefl_jupyter/bin/exe
ENV DOCKER=true

# Change user to default jupyter user (from jupyter/scipy-notebook Dockerfile)
USER $NB_UID

# Install packages with pip
COPY --chown=${NB_UID}:${NB_GID} requirements.txt /tmp/
RUN pip3 install --quiet --no-cache-dir --requirement /tmp/requirements.txt && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN pip3 install numpy --upgrade

RUN mkdir Files

COPY notebooks/homework/* /home/jovyan/gnssrefl_jupyter/notebooks/homework/
COPY notebooks/use-cases/* /home/jovyan/gnssrefl_jupyter/notebooks/use-cases/

# Start the jupyter notebook upon executuion
ENTRYPOINT ["jupyter", "notebook", "--ip=*", "--allow-root", "--notebook-dir=/home/jovyan/gnssrefl_jupyter"]
