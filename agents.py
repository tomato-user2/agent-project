from langgraph.graph import StateGraph, END
from search import duckduckgo_search
import ollama
import asyncio
import json

# Node 1: Extract books from user input
async def extract_books_node(state):
    user_input = state.get("user_input", "")
    prompt = (
        f"Extract all book titles and authors from the following text. "
        "If an author is missing, fill it in using your knowledge. "
        "Output as a JSON list of dicts like this: "
        "[{{'title': '...', 'author': '...'}}, ...].\n\n"
        f"User input: {user_input}"
    )
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])

    try:
        books = json.loads(response['message']['content'])
    except Exception:
        books = []

    return {"extracted_books": books}

# Node 2: Recommend similar books using DuckDuckGo search
async def recommend_books_node(state):
    extracted_books = state.get("extracted_books", [])
    reasoning_steps = []
    recommended_books = []

    for book in extracted_books:
        query = f"Books similar to '{book['title']}' by {book['author']}"
        reasoning_steps.append(f"Searching for: {query}")
        search_results = await duckduckgo_search(query)

        for res in search_results:
            recommended_books.append({
                "title": res["title"],
                "link": res["link"],
                "snippet": res["snippet"]
            })
            reasoning_steps.append(f"- Found: {res['title']} ({res['link']})")

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
