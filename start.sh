#!/bin/bash

export HOME=/home/ollama
export OLLAMA_HOME=/home/ollama/.ollama

echo "Starting Ollama server..."
OLLAMA_HOST=0.0.0.0:7860 ollama serve &

# Wait a bit for the server to start
echo "Waiting for Ollama server to start..."
sleep 5

wait_for_ollama() {
    while ! ollama list >/dev/null 2>&1; do
        echo "Waiting for Ollama model API to be ready..."
        sleep 5
    done
    echo "Ollama model API is ready!"
}

wait_for_ollama

echo "Ollama is ready, waiting a few more seconds to stabilize..."
sleep 5

echo "Pulling llama3.2:1b model..."
ollama pull llama3.2:1b

echo "All models pulled successfully!"
echo "Ollama is running on http://0.0.0.0:7860"

echo "Starting your app..."
python3 app.py
