FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system tools and curl
RUN apt-get update && \
    apt-get install -y curl bash && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | bash

# Confirm installation
RUN ollama --version

# Create ollama user and home dir
RUN useradd -m -u 1000 ollama

# Set correct home and Ollama data directory
ENV HOME=/home/ollama
ENV OLLAMA_HOME=/home/ollama/.ollama

# Set working directory
WORKDIR /home/ollama/app

# Copy everything as ollama user
COPY --chown=ollama:ollama . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make start.sh executable
RUN chmod +x start.sh

# Switch to ollama user
USER ollama

# Expose ports
EXPOSE 7860 7861

# Run startup script
CMD ["bash", "./start.sh"]
