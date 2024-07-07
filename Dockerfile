FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    && apt-get clean

# Download and install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -u -p /opt/conda && \
    rm /tmp/miniconda.sh


ENV PATH=/opt/conda/bin:$PATH

# Install missing dependencies
RUN conda install -y chardet charset_normalizer archspec

# Copy environment.yml to the working directory
COPY environment.yaml /tmp/environment.yaml

# Create the Conda environment without plugins
RUN CONDA_NO_PLUGINS=true conda env create -f /tmp/environment.yaml

# Copy entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
