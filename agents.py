from langgraph.graph import StateGraph, END
from search import duckduckgo_search
import asyncio
import re
import json
import asyncio
import httpx
import os
import ast
import traceback
from huggingface_hub import InferenceClient

# Create a single shared client
# It will read your HUGGINGFACEHUB_API_TOKEN from the env for authentication
client = InferenceClient(token=os.getenv("HF_API_TOKEN"))

async def hf_chat(model: str, messages: list[dict]):
    loop = asyncio.get_running_loop()

    def _sync_call():
        # Ensure you have initialized the client with your HF_API_TOKEN
        return client.chat.completions.create(
            model=model,
            messages=messages,
            # you can pass generation params here too
            # temperature=0.7, max_tokens=512, ...
        )

    completion = await loop.run_in_executor(None, _sync_call)

    return {
        "message": {
            "role": completion.choices[0].message.role,
            "content": completion.choices[0].message.content
        }
    }

# Alias `chat` to your HF-backed version
chat = hf_chat

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

def extract_json_array(text: str):
    # Remove Markdown/HTML formatting
    text = re.sub(r"```(?:json)?\n?|</?(?:pre|code|p)>", "", text, flags=re.IGNORECASE)

    # Extract the first [...] block
    match = re.search(r"(\[\s*{.*?}\s*\])", text, re.DOTALL)
    if not match:
        return []
    
    json_str = match.group(1)

    # Try parsing as JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print("[extract_json_array] JSON decode error:", e)

        # Fallback: try ast.literal_eval
        try:
            return ast.literal_eval(json_str)
        except Exception as e2:
            print("[extract_json_array] literal_eval failed:", e2)
            return []

# Node 1: Extract books from user input
async def extract_books_node(state):
    try:
        print("[extract_books_node] 👉 enter")
        user_input = state.get("user_input", "")
        prompt = (
            "Extract all book titles and authors from the user input. Do not add books on your own, just take the user input."
            "If a book is mentioned but the author is missing, try to fill the missing author in using reasoning with your knowledge."
            "ONLY output a JSON list of dicts, like this:\n"
            '[{"title": "...", "author": "..."}, ...]\n'
            "Do not add any explanations, prefixes, or markdown. Just the JSON list.\n\n"
            f"User input: {user_input}"
        )
        print("[extract_books_node] Prompt sent to LLM:\n", prompt)

        response = await chat(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[{"role":"user","content": prompt}]
        )
        content = response["message"]["content"]

        print("[extract_books_node] Raw LLM response:\n", repr(content))
        print(f"[extract_books_node] Response type: {type(content)}, length: {len(content)}")

        # Remove code blocks or markdown again here just to be sure
        cleaned_content = re.sub(r"```(?:json)?\n?|</?(?:pre|code|p)>", "", content, flags=re.IGNORECASE).strip()
        print("[extract_books_node] Cleaned response:\n", repr(cleaned_content))

        books = []
        try:
            books = json.loads(cleaned_content)
            print("[extract_books_node] JSON parsed successfully.")
        except json.JSONDecodeError as e:
            print(f"[extract_books_node] JSONDecodeError: {e}")
            print("[extract_books_node] Attempting fallback parsing with ast.literal_eval.")
            try:
                books = ast.literal_eval(cleaned_content)
                print("[extract_books_node] Fallback parsing successful.")
            except Exception as e2:
                print(f"[extract_books_node] Fallback parsing failed: {e2}")
                print("[extract_books_node] Traceback:\n", traceback.format_exc())

        print("[extract_books_node] Extracted books:", books)
        print("[extract_books_node] 👈 exit with", {"extracted_books": books})
        return {"extracted_books": books}

    except Exception as e:
        print("[extract_books_node] ❌ exception:", repr(e))
        print("[extract_books_node] Traceback:\n", traceback.format_exc())
        raise

# Node 1.1 New Node: Complete missing authors
async def complete_authors_node(state):
    try:
        print("[complete_authors_node] 👉 enter")
        books = state.get("extracted_books", [])
        incomplete_books = [book for book in books if not book.get("author", "").strip()]

        if not incomplete_books:
            print("[complete_authors_node] No missing authors to complete.")
            return {"extracted_books": books}

        # Prepare prompt for LLM
        prompt = (
            "You are given a list of books with some missing authors. "
            "For each book, fill in the correct author using your knowledge. "
            "ONLY output a JSON list like this:\n"
            '[{"title": "...", "author": "..."}, ...]\n\n'
            "Do not add explanations, prefixes, or markdown. Only output the JSON list.\n\n"
            f"Books with missing authors:\n{json.dumps(incomplete_books, ensure_ascii=False)}"
        )

        print("[complete_authors_node] Prompt sent to LLM:\n", prompt)

        response = await chat(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response["message"]["content"]

        print("[complete_authors_node] Raw LLM response:\n", repr(content))
        print(f"[complete_authors_node] Response type: {type(content)}, length: {len(content)}")

        # Remove code blocks or markdown
        cleaned_content = re.sub(r"```(?:json)?\n?|</?(?:pre|code|p)>", "", content, flags=re.IGNORECASE).strip()
        print("[complete_authors_node] Cleaned response:\n", repr(cleaned_content))

        completed_books_from_llm = []
        try:
            completed_books_from_llm = json.loads(cleaned_content)
            print("[complete_authors_node] JSON parsed successfully.")
        except json.JSONDecodeError as e:
            print(f"[complete_authors_node] JSONDecodeError: {e}")
            print("[complete_authors_node] Attempting fallback parsing with ast.literal_eval.")
            try:
                completed_books_from_llm = ast.literal_eval(cleaned_content)
                print("[complete_authors_node] Fallback parsing successful.")
            except Exception as e2:
                print(f"[complete_authors_node] Fallback parsing failed: {e2}")
                print("[complete_authors_node] Traceback:\n", traceback.format_exc())

        # Merge back into the full book list
        title_to_author = {book["title"]: book.get("author", "Unknown") for book in completed_books_from_llm}
        completed_books = []
        for book in books:
            title = book.get("title", "").strip()
            author = book.get("author", "").strip()
            if not author:
                # Fill from LLM result or fallback to DuckDuckGo
                author = title_to_author.get(title, "").strip()
                if not author:
                    # DuckDuckGo fallback if still missing
                    query = f"{title} book author"
                    print(f"[complete_authors_node] Searching DuckDuckGo for author: {query}")
                    search_results = await duckduckgo_search(query)

                    found_author = "Unknown"
                    if search_results:
                        for res in search_results:
                            snippet = res.get("snippet", "")
                            title_text = res.get("title", "")
                            match = re.search(r"by ([A-Z][a-z]+(?: [A-Z][a-z]+)*)", snippet + " " + title_text)
                            if match:
                                found_author = match.group(1)
                                print(f"[complete_authors_node] Found author '{found_author}' for book '{title}'")
                                break
                    author = found_author

            completed_books.append({
                "title": title,
                "author": author
            })

        print("[complete_authors_node] Completed books list:", completed_books)
        return {"extracted_books": completed_books}

    except Exception as e:
        print("[complete_authors_node] ❌ exception:", repr(e))
        print("[complete_authors_node] Traceback:\n", traceback.format_exc())
        raise

# Node 2
async def recommend_books_node(state):
    try:
        print("[recommend_books_node] 👉 enter")
        extracted_books = state.get("extracted_books", [])
        reasoning_steps = []
        recommended_books = []

        print("[recommend_books_node] Extracted books:", extracted_books)
        # await logger.log(f"[recommend_books_node] Extracted books: {extracted_books}")

        if not extracted_books:
            reasoning_steps.append("No books extracted from the input. Check if the extraction failed.")
            return {"recommendations": [], "reasoning": "\n".join(reasoning_steps)}

        for book in extracted_books:
            title = book.get("title", "")
            author = book.get("author", "")
            query = f"Books similar to '{title}' by {author}"
            reasoning_steps.append(f"Searching DuckDuckGo with query: {query}")

            print(f"[recommend_books_node] Searching with query: {query}")
            # await logger.log(f"Searching DuckDuckGo with query: {query}")

            search_results = await duckduckgo_search(query)

            if not search_results:
                reasoning_steps.append(f"No results found for: {query}")
                print(f"[recommend_books_node] No results found for query: {query}")
                # await logger.log(f"No results found for query: {query}")
                continue

            print(f"[recommend_books_node] Results for query '{query}': {search_results}")

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
        print("[recommend_books_node] 👈 exit with", {"recommendations": recommended_books, "reasoning": "\n".join(reasoning_steps)})
        return {
            "recommendations": recommended_books,
            "reasoning": "\n".join(reasoning_steps)
        }
    
    except Exception as e:
        print("[extract_books_node] ❌ exception:", repr(e))
        raise

# Node 3: Reason about the search results and generate recommendations

async def reasoning_node(state):
    try:
        recommendations = state.get("recommendations", [])
        initial_reasoning = state.get("reasoning", "")

        if not recommendations:
            final_reasoning = initial_reasoning + "\nNo recommendations found to reason about."
            print("[reasoning_node] No recommendations to process.")
            return {"final_recommendations": [], "final_reasoning": final_reasoning}

        # Format recommendations as input for the LLM
        recommendations_text = "\n".join(
            [f"Title: {rec['title']}\nLink: {rec['link']}\nSnippet: {rec['snippet']}\n" for rec in recommendations]
        )

        prompt = (
            "You are a helpful book recommendation expert. You are given a web search result. "
            "Analyze it and select the most relevant book recommendations. Explain why you recommend each book. "
            "Do not recommend the same books from the user input!"
            "Output only a JSON list like this:\n"
            '[{"title": "...", "reason": "...", "link": "..."}, ...]\n\n'
            "Do not add any explanations, comments, or extra text. Only output the JSON list.\n\n"
            f"Books found from search:\n{recommendations_text}"
        )

        print("[reasoning_node] Prompt sent to LLM:\n", prompt)

        response = await chat(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[{"role":"user","content": prompt}]
        )

        content = response['message']['content']

        print("[reasoning_node] Raw LLM response:\n", repr(content))
        print(f"[reasoning_node] Response type: {type(content)}, length: {len(content)}")

        # Clean the content from code blocks, markdown, etc.
        cleaned_content = re.sub(r"```(?:json)?\n?|</?(?:pre|code|p)>", "", content, flags=re.IGNORECASE).strip()
        print("[reasoning_node] Cleaned response:\n", repr(cleaned_content))

        final_recommendations = []
        try:
            final_recommendations = json.loads(cleaned_content)
            print("[reasoning_node] JSON parsed successfully.")
        except json.JSONDecodeError as e:
            print(f"[reasoning_node] JSONDecodeError: {e}")
            print("[reasoning_node] Attempting fallback parsing with ast.literal_eval.")
            try:
                final_recommendations = ast.literal_eval(cleaned_content)
                print("[reasoning_node] Fallback parsing successful.")
            except Exception as e2:
                print(f"[reasoning_node] Fallback parsing failed: {e2}")
                print("[reasoning_node] Traceback:\n", traceback.format_exc())

        # Compose final reasoning combining initial and LLM results
        final_reasoning = initial_reasoning + "\n\nFinal reasoning:\n"
        for rec in final_recommendations:
            final_reasoning += f"✅ Recommended: {rec.get('title', 'Unknown')} - {rec.get('reason', 'No reason provided.')}\n"

        print("[reasoning_node] Final recommendations extracted:", final_recommendations)
        print("[reasoning_node] Final reasoning:\n", final_reasoning)

        return {
            "final_recommendations": final_recommendations,
            "final_reasoning": final_reasoning
        }

    except Exception as e:
        print("[reasoning_node] ❌ exception:", repr(e))
        print("[reasoning_node] Traceback:\n", traceback.format_exc())
        raise



# Build the graph
def build_graph():
    graph = StateGraph(dict)

    graph.add_node("extract_books", extract_books_node)
    graph.add_node("complete_authors", complete_authors_node)  # <-- New node
    graph.add_node("recommend_books", recommend_books_node)
    graph.add_node("reasoning", reasoning_node)

    # Define edges
    graph.add_edge("extract_books", "complete_authors")  # Modified
    graph.add_edge("complete_authors", "recommend_books")  # Modified
    graph.add_edge("recommend_books", "reasoning")
    graph.add_edge("reasoning", END)

    graph.set_entry_point("extract_books")
    return graph.compile()
