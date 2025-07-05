---
title: Ollama Chat API
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Ollama Chat API

This space runs Ollama with llama3.2:1b and mxbai-embed-large models for chat and embeddings.

## API Endpoints

- `GET /api/tags` - List available models
- `POST /api/generate` - Generate text
- `POST /api/embeddings` - Generate embeddings

## Models Available

- `llama3.2:1b` - Chat model
- `mxbai-embed-large` - Embedding model

The API is compatible with Ollama's standard API format.