FROM nvcr.io/nvidia/jax:23.10-py3

# Create user
ARG UID
ARG MYUSER
RUN useradd -u $UID --create-home ${MYUSER}
USER ${MYUSER}

# default workdir
WORKDIR /home/${MYUSER}/
COPY --chown=${MYUSER} --chmod=765 . .

USER root

# install tmux
RUN apt-get update && \
    apt-get install -y tmux

#jaxmarl from source if needed, all the requirements
RUN pip install moviepy==1.0.3 && \
    pip install wandb==0.18.5 && \
    pip install -e . && \
    # pip install --upgrade pip && \
    # pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    # pip install matplotlib && \
    # pip install tensorflow_probability && \
    # pip install tqdm && \
    # pip install distrax && \
    pip install hydra-core --upgrade && \
    pip install hydra_colorlog --upgrade 

USER ${MYUSER}

#disabling preallocation
RUN export XLA_PYTHON_CLIENT_PREALLOCATE=false
#safety measures
RUN export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 
RUN export TF_FORCE_GPU_ALLOW_GROWTH=true

# Uncomment below if you want jupyter 
RUN pip install jupyterlab

RUN git config --global --add safe.directory /home/${MYUSER} && \
    git config --global core.autocrlf input