from smolagents.tools import tool
import re
import requests
import logging

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
    Given a list of input books (title + author), suggest other similar books.

    Args:
        book_list: A list of dicts with keys 'title' and 'author'.

    Returns:
        A list of recommendations, each a dict with:
            - 'title': the recommended book's title
            - 'author': the recommended book's author (if known)
            - 'reason': a short reason why it was recommended
    """
    # The actual recommendation logic will be handled by the agent via LLM reasoning.
    return []