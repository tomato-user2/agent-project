from smolagents.tools import tool
import re
import requests
import logging
from bs4 import BeautifulSoup

def duckduckgo_search_snippets(query: str, max_snippets: int = 3) -> list:
    """
    Performs a DuckDuckGo search and extracts text snippets from the results.
    """
    logging.debug(f"[duckduckgo_search_snippets] Query: {query}")
    try:
        resp = requests.get("https://lite.duckduckgo.com/lite/", params={"q": query}, timeout=10)
        if not resp.ok:
            return [f"[Error] Search failed for query: {query}"]

        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.find_all("a", class_="result-link")[:max_snippets]
        snippets = []

        for link in results:
            snippet = link.get_text(strip=True)
            if snippet:
                snippets.append(snippet)

        return snippets or ["[No results found]"]
    except Exception as e:
        logging.error(f"[duckduckgo_search_snippets] Failed: {e}")
        return [f"Search error: {e}"]

@tool
def extract_books(text: str) -> list:
    """
    Use LLM reasoning to extract a list of books from the user's input.

    Args:
        text: Free-form user input describing books they liked.

    Returns:
        A list of dictionaries, each with keys:
            - 'title': the title of the book
            - 'author': the author of the book (if known)
    """
    # We’ll delegate extraction to the LLM itself.
    # This is just a placeholder tool schema, the model will generate the response
    return []

@tool
def search_web(query: str) -> str:
    """
    Perform a web search for information about books, authors, or topics.

    Args:
        query (str): The search query to look up.

    Returns:
        str: A text snippet or summary from the search results.
    """
    logging.debug(f"[search_web] Searching: {query}")
    try:
        resp = requests.get("https://lite.duckduckgo.com/lite/", params={"q": query}, timeout=10)
        if resp.ok:
            return resp.text[:1000]
    except Exception as e:
        logging.error(f"[search_web] Failed: {e}")
        return f"Search error: {e}"

@tool
def recommend_similar_books(book_list: list) -> list:
    """
    Given a list of input books (title + author), search the web and suggest other similar books.

    Args:
        book_list: A list of dicts with keys 'title' and 'author'.

    Returns:
        A list of recommendations, each a dict with:
            - 'title': recommended book's title
            - 'author': author (if known)
            - 'reason': why it was recommended
    """
    from smolagents import call_llm  # ensure this is imported for calling your LLM

    all_snippets = ""
    for book in book_list:
        query = f"Books similar to '{book['title']}' by {book.get('author', 'unknown author')}"
        snippets = duckduckgo_search_snippets(query)
        all_snippets += f"### Search results for {book['title']}:\n" + "\n".join(snippets) + "\n\n"

    # Now ask the LLM to analyze snippets and generate recommendations
    prompt = f"""You are a book recommendation assistant.

Here are search results for books similar to some user favorites:

{all_snippets}

Based on these, recommend 3 books. For each, include:
- title
- author (if known)
- reason for recommendation (based on search result info)

Return as a list of JSON objects.
"""
    response = call_llm(prompt, model="ollama/llama3", api_base="http://localhost:11434")

    try:
        # Try to parse list of dicts from response (defensive parsing)
        import json
        recommendations = json.loads(response)
        if isinstance(recommendations, list):
            return recommendations
    except Exception as e:
        logging.error(f"Failed to parse LLM response: {e}")

    return [{"title": "Unknown", "author": "Unknown", "reason": "Failed to parse LLM output."}]
