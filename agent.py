from smolagents import ToolCallingAgent, LiteLLMModel, DuckDuckGoSearchTool
from tools import extract_books, recommend_similar_books

# Local Llama3 via Ollama
model = LiteLLMModel(
    model_id="ollama/llama3",
    api_base="http://localhost:11434"
)

agent = ToolCallingAgent(
    tools=[extract_books, DuckDuckGoSearchTool(), recommend_similar_books],
    model=model,
    stream_outputs=True,   # optional real-time output
)