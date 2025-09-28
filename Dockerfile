# Start from your TF 1.13 + Python 3.6 base
FROM python:3.6-slim

WORKDIR /workspace

# Upgrade pip/setuptools/wheel
RUN pip install --upgrade pip setuptools wheel

# Copy your pre-downloaded TF 1.13.1 wheel
COPY wheels/ /tmp/wheels/

# Install TensorFlow + pinned deps
RUN pip install --no-cache-dir \
    /tmp/wheels/tf_nightly-1.13.1-cp36-cp36m-linux_x86_64.whl \
    tf-slim==1.1.0 \
    h5py==2.10.0 \
    numpy==1.16.6 \
    tensorflow-estimator==1.13.0 \
    keras-applications==1.0.7 \
    keras-preprocessing==1.0.9

# --- Add Slim models repo at r1.13.0 ---
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/tensorflow/models.git /opt/models && \
    cd /opt/models && git checkout r1.13.0

# Add Slim to PYTHONPATH
ENV PYTHONPATH=$PYTHONPATH:/opt/models/research/slim

# Keep container running (optional for dev/debug)
CMD ["tail", "-f", "/dev/null"]





# # lighter than tf/tf:1.13 base; use py3.6 to match wheel
# FROM python:3.6-slim

# WORKDIR /workspace

# # Upgrade pip/setuptools/wheel
# RUN pip install --upgrade pip setuptools wheel

# # # Install TensorFlow without AVX requirements using community wheel
# # RUN pip install --no-cache-dir \
# #     https://github.com/Tzeny/tensorflowbuilds/raw/master/tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl \
# #     h5py==2.10.0 \
# #     numpy==1.16.6

# # Install TensorFlow without AVX requirements using community wheel
# # AND add TensorFlow Slim
# RUN pip install --no-cache-dir \
#     https://github.com/Tzeny/tensorflowbuilds/raw/master/tensorflow-1.13.1-cp36-cp36m-linux_x86_64.whl \
#     tf-slim==1.1.0 \
#     h5py==2.10.0 \
#     numpy==1.16.6

# # # Install standard tensorflow-cpu (may work without AVX)
# # RUN pip install --no-cache-dir \
# #     tensorflow-cpu==1.13.1 \
# #     h5py==2.10.0 \
# #     numpy==1.16.6

# # Keep container running
# CMD ["tail", "-f", "/dev/null"]

# # # lighter than tf/tf:1.13 base; use py3.6 to match wheel
# # FROM python:3.6-slim
   

# # WORKDIR /workspace

# # # Upgrade pip/setuptools/wheel
# # RUN pip install --upgrade pip setuptools wheel

# # # Copy wheels into image
# # COPY wheels/ /tmp/wheels/

# # # Install from local wheel folder, skip online fetching
# # RUN pip install --no-cache-dir \
# #     --find-links=/tmp/wheels \
# #     tensorflow==1.13.1 \
# #     h5py==2.10.0 \
# #     numpy==1.16.6

# # # Keep container running
# # CMD ["tail", "-f", "/dev/null"]
