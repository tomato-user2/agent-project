from smolagents.tools import tool
import re
import requests
import logging
from bs4 import BeautifulSoup
import json

# Optional: Configure logging
logging.basicConfig(level=logging.INFO)

def extract_json_array(text):
    """
    Extract the first JSON array from a text string,
    ignoring any leading or trailing non-JSON text.
    """
    # Try to find a JSON array with a regex (non-greedy)
    match = re.search(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error on regex candidate: {e}")
    # Fallback: take substring from first [ to last ]
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error on fallback candidate: {e}")
    raise ValueError("No valid JSON array found in text")

def call_llm(prompt: str, model: str = "ollama/llama3", api_base: str = "http://localhost:11434") -> str:
    """
    Calls a local LLM (e.g., via Ollama HTTP API) with a prompt.

    Args:
        prompt (str): The input prompt for the LLM.
        model (str): The model to use, e.g., "ollama/llama3".
        api_base (str): The base URL of the model endpoint.

    Returns:
        str: The generated text from the model.
    """
    try:
        response = requests.post(
            f"{api_base}/api/generate",
            json={"model": model.replace("ollama/", ""), "prompt": prompt, "stream": False},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        logging.error(f"[call_llm] Failed to call LLM: {e}")
        return ""


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
    Extracts book titles and authors from user input.

    Args:
        text: Free-form user input describing books they liked.

    Returns:
        A list of dictionaries with:
            - 'title': Title of the book
            - 'author': Author of the book (if known or added via search)
    """
    logging.debug(f"[extract_books] Input: {text}")

    prompt = f"""Extract all books and their authors from this text:

{text}

Return a JSON list like this:
[
  {{
    "title": "Book Title",
    "author": "Author Name"
  }},
  ...
]
"""

    try:
        response = call_llm(prompt)
        # Try to extract JSON block from response
        match = re.search(r"\[\s*{.*?}\s*\]", response, re.DOTALL)
        if match:
            cleaned_json = match.group(0)
            parsed = json.loads(cleaned_json)
            if isinstance(parsed, list):
                return parsed
        raise ValueError("No valid JSON array found")
    except Exception as e:
        logging.error(f"[extract_books] Failed: {e}\nLLM response: {response}")
        return [{"title": "Unknown", "author": "Unknown"}]


@tool
def recommend_similar_books(book_list: list[dict]) -> list[dict]:
    """
    Given a list of books, search the web and suggest similar books.

    Args:
        book_list (list of dict): A list of dictionaries, each with:
            - 'title' (str): The title of the book.
            - 'author' (str): The author of the book.

    Returns:
        list of dict: A list of recommended books. Each dictionary contains:
            - 'title' (str): Title of the recommended book.
            - 'author' (str): Author of the recommended book.
            - 'reason' (str): Explanation of why it was recommended.
    """

    all_snippets = ""
    for book in book_list:
        query = f"Books similar to '{book['title']}' by {book.get('author', 'unknown author')}"
        snippets = duckduckgo_search_snippets(query)
        all_snippets += f"### Search results for {book['title']}:\n" + "\n".join(snippets) + "\n\n"

    prompt = f"""You are a book recommendation assistant.

Here are search results for books similar to user favorites:

{all_snippets}

Based on these, recommend 3 books. For each, include:
- title
- author (if known)
- reason for recommendation (based on search result info)

ONLY Return as a JSON list like this:
[
  {{
    "title": "...",
    "author": "...",
    "reason": "..."
  }},
  ...
]
NO explanations before or after the JSON block.
"""

    response = call_llm(prompt, model="ollama/llama3", api_base="http://localhost:11434")

    try:
        recommendations = extract_json_array(response)
        if not isinstance(recommendations, list):
            raise ValueError("Parsed JSON is not a list")
        return {
            "search_snippets": all_snippets,
            "llm_prompt": prompt,
            "llm_response": response,
            "recommendations": recommendations,
                }

    except Exception as e:
        logging.error(f"[recommend_similar_books] Failed to parse LLM response: {e}\nResponse was:\n{response}")
        # Return a safe fallback to avoid crashing
        return [{"title": "Unknown", "author": "Unknown", "reason": "Failed to parse LLM output."}]
