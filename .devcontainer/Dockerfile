# Change this line from python:3.9 to python:3.10
FROM python:3.10-slim

# avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

# Install OS deps for matplotlib, torch, and building packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      libgl1 \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Create & use workspace directory
WORKDIR /workspace

# Copy in requirements and install Python packages
COPY requirements.txt /workspace/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Default to bash
CMD ["bash"]