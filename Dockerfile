FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Install system dependencies for HDF5
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install h5py and any other Python dependencies
RUN pip install --upgrade pip && \
    pip install h5py

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/app





