# Use the official PyTorch image as the base image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set environment variable to make Python output unbuffered
ENV PYTHONUNBUFFERED=1

# Install additional dependencies via apt-get
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Ensure conda is in the PATH
ENV PATH=/opt/conda/bin:$PATH

# Update Conda
RUN conda update -n base -c defaults conda

# Copy environment.yml to the working directory
COPY environment.yml /tmp/environment.yml

# Create the Conda environment using the environment.yml file
RUN conda env create -f /tmp/environment.yml || (cat /tmp/environment.yml && exit 1)

# Print list of environments for debugging
RUN conda env list

# Add conda initialization to .bashrc
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Activate the environment
RUN echo "conda activate photoens" >> ~/.bashrc

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "photoens", "/bin/bash", "-c"]

# Copy entrypoint script to the appropriate location
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/rlenh/src /app/rlenh/experiments /app/rlenh/dataset

# Copy only the src directory
COPY src /app/rlenh/src/

WORKDIR /app/rlenh

# Set the entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Expose port 6006 for TensorBoard
EXPOSE 6006

# Set the default command
CMD ["python", "src/train.py", "solos", "experiments/configs/hyperparameters.yaml", "experiments/configs/config.yaml"]