#!/bin/bash

echo "Starting Ollama server..."
OLLAMA_HOST=0.0.0.0:7860 ollama serve &

# Wait for Ollama to be ready
echo "Waiting for Ollama server to start..."
sleep 15

wait_for_ollama() {
    while ! curl -s http://localhost:7860/api/tags > /dev/null; do
        echo "Waiting for Ollama to be ready..."
        sleep 5
    done
    echo "Ollama is ready!"
}

wait_for_ollama

echo "Pulling llama3.2:1b model..."
ollama pull llama3.2:1b

echo "Pulling mxbai-embed-large model..."
ollama pull mxbai-embed-large

echo "All models pulled successfully!"
echo "Ollama is running on http://0.0.0.0:7860"

# Start your app (replace app.py with your actual script name)
echo "Starting your app..."
python3 app.py
