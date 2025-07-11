FROM tensorflow/tensorflow:2.19.0-gpu

ENV DEBIAN_FRONTEND=noninteractive

# Set venv path inside /app (which is typically bind-mounted)
ENV VENV_PATH=/workspace/venv
ENV PATH="$VENV_PATH/bin:$PATH"

# Install Python and tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    ca-certificates \
    curl \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create working directory (usually bind-mounted to your host code)
WORKDIR /workspace/app

# Create the virtual environment in /app/venv
# Then install Python packages there
COPY requirements.txt .
RUN python3 -m venv $VENV_PATH && \
    $VENV_PATH/bin/pip install --upgrade pip && \
    $VENV_PATH/bin/pip install -r requirements.txt

RUN $VENV_PATH/bin/pip uninstall tensorflow -y && \
    $VENV_PATH/bin/pip install tensorflow[and-cuda]

CMD ["bash"]
