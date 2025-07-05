#!/bin/bash
# start.sh for Hugging Face Spaces

echo "Starting Ollama server..."

# Start Ollama server in background with explicit host:port
OLLAMA_HOST=0.0.0.0:7860 ollama serve &

# Wait for server to start
echo "Waiting for Ollama server to start..."
sleep 15

# Function to check if Ollama is ready
wait_for_ollama() {
    while ! curl -s http://localhost:7860/api/tags > /dev/null; do
        echo "Waiting for Ollama to be ready..."
        sleep 5
    done
    echo "Ollama is ready!"
}

wait_for_ollama

# Pull required models
echo "Pulling llama3.2:1b model..."
ollama pull llama3.2:1b

echo "Pulling mxbai-embed-large model..."
ollama pull mxbai-embed-large

echo "All models pulled successfully!"
echo "Ollama is running on http://0.0.0.0:7860"

echo "Starting Gradio app..."
python3 app.py

# Keep container running and show logs
tail -f /dev/null