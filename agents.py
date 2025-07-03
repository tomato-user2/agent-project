from langgraph.graph import StateGraph, END
from search import duckduckgo_search
import ollama
import asyncio
import re
import json
import asyncio
import ast

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
    # Try to extract the list from the entire text by searching for a list literal
    pattern = r"(\[.*\])"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if not matches:
        return []

    for candidate in matches:
        try:
            # Safely evaluate the python-like list/dict string
            data = ast.literal_eval(candidate)
            # Convert to JSON string and parse to verify
            json_str = json.dumps(data)
            return json.loads(json_str)
        except Exception as e:
            print(f"ast.literal_eval/json error: {e}")
            continue
    return []

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
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    content = response['message']['content']

    print("[extract_books_node] LLM raw response:", content)
    await logger.log(f"[extract_books_node] LLM response: {content}")

    books = extract_json_array(content)

    if not books:
        await logger.log("[extract_books_node] Failed to extract valid book list from LLM response.")
    else:
        await logger.log(f"[extract_books_node] Extracted books: {books}")

    print("[extract_books_node] Extracted books:", books)

    return {"extracted_books": books}

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

        search_results = await duckduckgo_search(query)

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


# Build the graph
def build_graph():
    graph = StateGraph(dict)

    graph.add_node("extract_books", extract_books_node)
    graph.add_node("recommend_books", recommend_books_node)

    graph.add_edge("extract_books", "recommend_books")
    graph.add_edge("recommend_books", END)

    graph.set_entry_point("extract_books")
    return graph.compile()
