# agents.py

from langgraph.graph import StateGraph, END
from search import duckduckgo_search
from huggingface_hub import InferenceClient
import asyncio
import re
import json
import os

# Load your Hugging Face token from environment variables
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Initialize the Hugging Face inference client
# Replace 'meta-llama/Meta-Llama-3-8b-chat-hf' with your preferred model
hf_client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8b-chat-hf",
    token=HF_API_TOKEN,
)

class AsyncLogger:
    def __init__(self):
        self._log = []
        self._lock = asyncio.Lock()
    
    async def log(self, message):
        async with self._lock:
            self._log.append(message)
    
    async def get_log(self):
        async with self._lock:
            return "\n".join(self._log)
    
    async def clear(self):
        async with self._lock:
            self._log.clear()

logger = AsyncLogger()

def extract_json_array(text):
    # Extract JSON block from anywhere in the text
    pattern = r"(\[.*?\])"  # non-greedy match to get the smallest bracketed block
    matches = re.findall(pattern, text, flags=re.DOTALL)

    for candidate in matches:
        try:
            # Attempt to load as JSON
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            print(f"json.loads error: {e}")
            continue

    return []

# Helper to query Hugging Face
async def hf_chat(prompt: str) -> str:
    print("[hf_chat] Sending prompt to Hugging Face Inference API")
    response = hf_client.text_generation(
        prompt=prompt,
        max_new_tokens=512,
        temperature=0.7,
        stop_sequences=["\n\n"],
    )
    return response.strip()

# Node 1: Extract books from user input
async def extract_books_node(state):
    await logger.clear()
    user_input = state.get("user_input", "")
    prompt = (
        "Extract all book titles and authors from the following text. "
        "If an author is missing, fill it in using your knowledge. "
        "Output only a JSON list of dicts like this:\n"
        '[{"title": "...", "author": "..."}, ...]\n\n'
        f"User input: {user_input}"
    )
    content = await hf_chat(prompt)

    print("[extract_books_node] LLM raw response:", content)
    await logger.log(f"[extract_books_node] LLM response: {content}")

    books = extract_json_array(content)

    if not books:
        await logger.log("[extract_books_node] Failed to extract valid book list from LLM response.")
    else:
        await logger.log(f"[extract_books_node] Extracted books: {books}")

    print("[extract_books_node] Extracted books:", books)

    return {"extracted_books": books}

# Node 2: Search for similar books
async def recommend_books_node(state):
    extracted_books = state.get("extracted_books", [])
    reasoning_steps = []
    recommended_books = []

    print("[recommend_books_node] Extracted books:", extracted_books)
    await logger.log(f"[recommend_books_node] Extracted books: {extracted_books}")

    if not extracted_books:
        reasoning_steps.append("No books extracted from the input. Check if the extraction failed.")
        return {"recommendations": [], "reasoning": "\n".join(reasoning_steps)}

    for book in extracted_books:
        title = book.get("title", "")
        author = book.get("author", "")
        query = f"Books similar to '{title}' by {author}"
        reasoning_steps.append(f"Searching DuckDuckGo with query: {query}")

        print(f"[recommend_books_node] Searching with query: {query}")
        await logger.log(f"Searching DuckDuckGo with query: {query}")

        search_results = await duckduckgo_search(query, logger=logger)

        if not search_results:
            reasoning_steps.append(f"No results found for: {query}")
            print(f"[recommend_books_node] No results found for query: {query}")
            await logger.log(f"No results found for query: {query}")
            continue

        print(f"[recommend_books_node] Results for query '{query}': {search_results}")
        await logger.log(f"Results for query '{query}': {search_results}")

        for res in search_results:
            recommended_books.append({
                "title": res.get("title", "No Title"),
                "link": res.get("link", ""),
                "snippet": res.get("snippet", "")
            })
            reasoning_steps.append(f"✅ Found: {res.get('title', 'No Title')} ({res.get('link', '')})")

    if not recommended_books:
        reasoning_steps.append("No recommendations found across all queries.")

    print("[recommend_books_node] Final recommendations:", recommended_books)
    await logger.log(f"Final recommendations: {recommended_books}")

    return {
        "recommendations": recommended_books,
        "reasoning": "\n".join(reasoning_steps)
    }

# Node 3: Reason over the search results
async def reasoning_node(state):
    recommendations = state.get("recommendations", [])
    initial_reasoning = state.get("reasoning", "")
    
    if not recommendations:
        final_reasoning = initial_reasoning + "\nNo recommendations found to reason about."
        return {"final_recommendations": [], "reasoning": {"final_recommendations": [], "final_reasoning": final_reasoning}}

    # Format recommendations for LLM
    recommendations_text = "\n".join(
        [f"Title: {rec['title']}\nLink: {rec['link']}\nSnippet: {rec['snippet']}\n" for rec in recommendations]
    )
    
    prompt = (
        "You are a helpful book recommendation expert. You are given a web search result. "
        "Analyze it and select the most relevant book recommendations. Explain why you recommend each book. "
        "Output only a JSON list like this:\n"
        '[{"title": "...", "reason": "...", "link": "..."}, ...]\n\n'
        "Do not add any explanations, comments, or extra text. Only output the JSON list.\n\n"
        f"Books found from search:\n{recommendations_text}"
    )

    content = await hf_chat(prompt)

    print("[reasoning_node] LLM raw response:", content)
    await logger.log(f"[reasoning_node] LLM response: {content}")

    # Extract JSON list
    final_recommendations = extract_json_array(content)

    if not final_recommendations:
        await logger.log("[reasoning_node] Failed to extract final recommendations from LLM response.")
    else:
        await logger.log(f"[reasoning_node] Final recommendations: {final_recommendations}")

    # Build final reasoning text
    final_reasoning = initial_reasoning + "\n\nFinal reasoning:\n"
    for rec in final_recommendations:
        final_reasoning += f"✅ Recommended: {rec.get('title', 'Unknown')} - {rec.get('reason', 'No reason provided.')}\n"

    print("[reasoning_node] Final recommendations extracted:", final_recommendations)
    print("[reasoning_node] Final reasoning:\n", final_reasoning)
    await logger.log(f"[reasoning_node] Final recommendations extracted: {final_recommendations}")
    await logger.log(f"[reasoning_node] Final reasoning:\n{final_reasoning}")

    return {
        "final_recommendations": final_recommendations,
        "reasoning": {"final_recommendations": final_recommendations, "final_reasoning": final_reasoning}
    }

# Build the graph
def build_graph():
    graph = StateGraph(dict)

    graph.add_node("extract_books", extract_books_node)
    graph.add_node("recommend_books", recommend_books_node)
    graph.add_node("reasoning", reasoning_node)

    # Define edges
    graph.add_edge("extract_books", "recommend_books")
    graph.add_edge("recommend_books", "reasoning")
    graph.add_edge("reasoning", END)

    graph.set_entry_point("extract_books")
    return graph.compile()