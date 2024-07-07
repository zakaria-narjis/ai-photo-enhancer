# Use the official PyTorch image as the base image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set environment variable to make Python output unbuffered
ENV PYTHONUNBUFFERED=1

# Install dependencies via apt-get
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

# Download and update Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -u -p /opt/conda && \
    rm /tmp/miniconda.sh

# Set environment variables
ENV PATH=/opt/conda/bin:$PATH
ENV CONDA_NO_PLUGINS=true

# Update Conda without plugins and install missing dependencies
RUN conda --no-plugins update -n base -c defaults conda && \
    conda --no-plugins install -y chardet charset_normalizer archspec

# Copy environment.yml to the working directory
COPY environment.yml /tmp/environment.yml

# Create the Conda environment using the environment.yml file
RUN conda env create -f /tmp/environment.yml

# Copy entrypoint script to the appropriate location
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

