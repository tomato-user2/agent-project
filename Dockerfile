FROM ollama/ollama:latest

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Create ollama user and directories
RUN useradd -m -u 1000 ollama
RUN mkdir -p /home/ollama/.ollama && chown -R ollama:ollama /home/ollama/.ollama

# Copy startup script
COPY --chown=ollama:ollama start.sh /home/ollama/start.sh
RUN chmod +x /home/ollama/start.sh

# Switch to ollama user
USER ollama
WORKDIR /home/ollama

# Set environment variables
ENV OLLAMA_HOST=0.0.0.0:7860
ENV HOME=/home/ollama

# Expose port
EXPOSE 7860

# Override entrypoint and run script
ENTRYPOINT []
CMD ["/bin/bash", "/home/ollama/start.sh"]