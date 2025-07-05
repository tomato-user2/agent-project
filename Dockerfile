FROM ollama/ollama:latest

# Install curl, Python, and pip
RUN apt-get update && apt-get install -y curl python3 python3-pip && rm -rf /var/lib/apt/lists/*

# Install your Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Create ollama user and directories
RUN useradd -m -u 1000 ollama
RUN mkdir -p /home/ollama/.ollama && chown -R ollama:ollama /home/ollama/.ollama

# Copy your app code
COPY --chown=ollama:ollama . /home/ollama/

# Switch to ollama user
USER ollama
WORKDIR /home/ollama

# Set environment variables
ENV OLLAMA_HOST=0.0.0.0:7860
ENV HOME=/home/ollama

# Expose ports
EXPOSE 7860 7861  # Add another port if your app serves an API/web UI

# Start Ollama + your app
ENTRYPOINT []
CMD ["/bin/bash", "/home/ollama/start.sh"]
