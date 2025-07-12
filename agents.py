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
        # Try to find any JSON array in the text
        match = re.search(r"(\[.*?\])", text, re.DOTALL)
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
            # Try to fix common JSON issues
            try:
                # Remove trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                # Fix unquoted keys
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)
                # Fix extra quotes around objects
                json_str = re.sub(r'"\s*{\s*"', '{"', json_str)
                json_str = re.sub(r'"\s*}\s*"', '"}', json_str)
                # Fix missing commas between objects
                json_str = re.sub(r'"\s*}\s*{', '"},{', json_str)
                # Fix extra quotes around individual objects in arrays
                json_str = re.sub(r'"\s*({[^}]+})\s*"', r'\1', json_str)
                return json.loads(json_str)
            except Exception as e3:
                print("[extract_json_array] JSON fixing failed:", e3)
                return []

def safe_json_parse(content: str, fallback_value=None):
    """Safely parse JSON content with multiple fallback strategies"""
    if fallback_value is None:
        fallback_value = []
    
    # Clean the content
    cleaned_content = re.sub(r"```(?:json)?\n?|</?(?:pre|code|p)>", "", content, flags=re.IGNORECASE).strip()
    
    # Try direct JSON parsing
    try:
        return json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        print(f"[safe_json_parse] JSONDecodeError: {e}")
        
        # Try to extract JSON array
        extracted = extract_json_array(cleaned_content)
        if extracted:
            return extracted
            
        # Try ast.literal_eval
        try:
            return ast.literal_eval(cleaned_content)
        except Exception as e2:
            print(f"[safe_json_parse] literal_eval failed: {e2}")
            
            # Try to fix common JSON issues
            try:
                # Remove trailing commas
                fixed_content = re.sub(r',\s*}', '}', cleaned_content)
                fixed_content = re.sub(r',\s*]', ']', fixed_content)
                # Fix unquoted keys
                fixed_content = re.sub(r'(\w+):', r'"\1":', fixed_content)
                # Fix single quotes to double quotes
                fixed_content = fixed_content.replace("'", '"')
                # Fix extra quotes around objects
                fixed_content = re.sub(r'"\s*{\s*"', '{"', fixed_content)
                fixed_content = re.sub(r'"\s*}\s*"', '"}', fixed_content)
                # Fix missing commas between objects
                fixed_content = re.sub(r'"\s*}\s*{', '"},{', fixed_content)
                # Fix extra quotes around individual objects in arrays
                fixed_content = re.sub(r'"\s*({[^}]+})\s*"', r'\1', fixed_content)
                return json.loads(fixed_content)
            except Exception as e3:
                print(f"[safe_json_parse] JSON fixing failed: {e3}")
                return fallback_value

def merge_state(current_state: dict, new_data: dict) -> dict:
    """Safely merge new data into current state, preserving existing data"""
    merged_state = current_state.copy()
    for key, value in new_data.items():
        if key in merged_state:
            # If both are lists, extend the current list
            if isinstance(merged_state[key], list) and isinstance(value, list):
                merged_state[key].extend(value)
            # If both are strings, concatenate them
            elif isinstance(merged_state[key], str) and isinstance(value, str):
                merged_state[key] += "\n" + value
            # Otherwise, overwrite
            else:
                merged_state[key] = value
        else:
            merged_state[key] = value
    return merged_state

# Node 1: Extract books from user input
async def extract_books_node(state):
    try:
        print("[extract_books_node] 👉 enter")
        user_input = state.get("user_input", "")
        prompt = (
            "Extract all book titles and authors from the user input. Do not add books on your own, just take the user input."
            "If a book is mentioned but the author is missing, try to fill the missing author in using reasoning with your knowledge."
            "IMPORTANT: Output ONLY a valid JSON array with this exact format:\n"
            '[{"title": "Book Title", "author": "Author Name"}]\n'
            "Rules:\n"
            "- Use double quotes for all strings\n"
            "- No trailing commas\n"
            "- No markdown formatting or code blocks\n"
            "- No explanations or extra text\n"
            "- If no books found, return empty array: []\n\n"
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

        # Use the new safe JSON parsing function
        books = safe_json_parse(content, fallback_value=[])
        
        # If parsing completely failed, try to extract book titles manually
        if not books and content:
            print("[extract_books_node] JSON parsing failed, attempting manual extraction")
            # Look for patterns like "title" or "book" in the content
            lines = content.split('\n')
            manual_books = []
            for line in lines:
                line = line.strip()
                if any(keyword in line.lower() for keyword in ['title', 'book', 'author']):
                    # Try to extract title and author from the line
                    title_match = re.search(r'"title":\s*"([^"]+)"', line)
                    author_match = re.search(r'"author":\s*"([^"]+)"', line)
                    if title_match:
                        title = title_match.group(1)
                        author = author_match.group(1) if author_match else "Unknown"
                        manual_books.append({"title": title, "author": author})
            
            if manual_books:
                books = manual_books
                print("[extract_books_node] Manual extraction successful:", books)
            else:
                # Last resort: try to extract from the specific malformed pattern we saw
                print("[extract_books_node] Attempting pattern-based extraction")
                # Look for patterns like "title": "Book Name"
                title_matches = re.findall(r'"title":\s*"([^"]+)"', content)
                author_matches = re.findall(r'"author":\s*"([^"]+)"', content)
                
                if title_matches:
                    for i, title in enumerate(title_matches):
                        author = author_matches[i] if i < len(author_matches) else "Unknown"
                        manual_books.append({"title": title, "author": author})
                    
                    if manual_books:
                        books = manual_books
                        print("[extract_books_node] Pattern-based extraction successful:", books)
        
        # Additional fix: if books is a list but contains malformed strings, try to fix them
        if isinstance(books, list) and books:
            print("[extract_books_node] Checking for malformed book entries...")
            fixed_books = []
            for book in books:
                if isinstance(book, str):
                    # Try to parse the string as JSON
                    try:
                        # Remove extra quotes around the object
                        cleaned_book = book.strip()
                        if cleaned_book.startswith('"') and cleaned_book.endswith('"'):
                            cleaned_book = cleaned_book[1:-1]
                        parsed_book = json.loads(cleaned_book)
                        if isinstance(parsed_book, dict) and parsed_book.get("title"):
                            fixed_books.append(parsed_book)
                    except:
                        # Try regex extraction as fallback
                        title_match = re.search(r'"title":\s*"([^"]+)"', book)
                        author_match = re.search(r'"author":\s*"([^"]+)"', book)
                        if title_match:
                            title = title_match.group(1)
                            author = author_match.group(1) if author_match else "Unknown"
                            fixed_books.append({"title": title, "author": author})
                elif isinstance(book, dict) and book.get("title"):
                    fixed_books.append(book)
            
            if fixed_books:
                books = fixed_books
                print("[extract_books_node] Fixed malformed book entries:", books)
        
        print("[extract_books_node] Parsed books:", books)

        # Ensure books is a list and each book has required fields
        if not isinstance(books, list):
            books = []
        
        # Validate and clean each book entry
        validated_books = []
        for book in books:
            if isinstance(book, dict):
                validated_book = {
                    "title": str(book.get("title", "")).strip(),
                    "author": str(book.get("author", "")).strip()
                }
                if validated_book["title"]:  # Only add if title is not empty
                    validated_books.append(validated_book)
        
        print("[extract_books_node] Validated books:", validated_books)
        print("[extract_books_node] 👈 exit with", {"extracted_books": validated_books})
        return {"extracted_books": validated_books}

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
            "IMPORTANT: Output ONLY a valid JSON array with this exact format:\n"
            '[{"title": "Book Title", "author": "Author Name"}]\n'
            "Rules:\n"
            "- Use double quotes for all strings\n"
            "- No trailing commas\n"
            "- No markdown formatting or code blocks\n"
            "- No explanations or extra text\n"
            "- Return all books, not just the ones with missing authors\n\n"
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

        # Use the new safe JSON parsing function
        completed_books_from_llm = safe_json_parse(content, fallback_value=[])
        print("[complete_authors_node] Parsed completed books:", completed_books_from_llm)

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

        # Validate the completed books
        validated_books = []
        for book in completed_books:
            if isinstance(book, dict):
                validated_book = {
                    "title": str(book.get("title", "")).strip(),
                    "author": str(book.get("author", "")).strip()
                }
                if validated_book["title"]:  # Only add if title is not empty
                    validated_books.append(validated_book)
        
        print("[complete_authors_node] Validated completed books:", validated_books)
        return {"extracted_books": validated_books}

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
            "IMPORTANT: Output ONLY a valid JSON array with this exact format:\n"
            '[{"title": "Book Title", "reason": "Why this book is recommended", "link": "URL"}]\n'
            "Rules:\n"
            "- Use double quotes for all strings\n"
            "- No trailing commas\n"
            "- No markdown formatting or code blocks\n"
            "- No explanations or extra text\n"
            "- If no good recommendations, return empty array: []\n\n"
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
        print("[reasoning_node] Response content (first 500 chars):", content[:500])

        # Use the new safe JSON parsing function
        final_recommendations = safe_json_parse(content, fallback_value=[])
        print("[reasoning_node] Parsed final recommendations:", final_recommendations)
        print("[reasoning_node] Type of final_recommendations:", type(final_recommendations))
        print("[reasoning_node] Length of final_recommendations:", len(final_recommendations) if isinstance(final_recommendations, list) else "Not a list")

        # Compose final reasoning combining initial and LLM results
        final_reasoning = initial_reasoning + "\n\nFinal reasoning:\n"
        for rec in final_recommendations:
            final_reasoning += f"✅ Recommended: {rec.get('title', 'Unknown')} - {rec.get('reason', 'No reason provided.')}\n"

        # Validate final recommendations
        validated_recommendations = []
        if isinstance(final_recommendations, list):
            for rec in final_recommendations:
                if isinstance(rec, dict):
                    validated_rec = {
                        "title": str(rec.get("title", "")).strip(),
                        "reason": str(rec.get("reason", "")).strip(),
                        "link": str(rec.get("link", "")).strip()
                    }
                    if validated_rec["title"]:  # Only add if title is not empty
                        validated_recommendations.append(validated_rec)
        
        print("[reasoning_node] Validated final recommendations:", validated_recommendations)
        print("[reasoning_node] Final reasoning:\n", final_reasoning)

        # Return the new state with our data
        result_state = {
            "final_recommendations": validated_recommendations,
            "final_reasoning": final_reasoning
        }
        
        print("[reasoning_node] Returning state with keys:", list(result_state.keys()))
        print("[reasoning_node] 👈 exit with", result_state)
        
        # Try returning as a dict to ensure proper state handling
        return dict(result_state)

    except Exception as e:
        print("[reasoning_node] ❌ exception:", repr(e))
        print("[reasoning_node] Traceback:\n", traceback.format_exc())
        # Return a safe fallback state instead of raising
        print("[reasoning_node] Returning fallback state due to exception")
        return {
            "final_recommendations": [],
            "final_reasoning": f"Error in reasoning node: {str(e)}"
        }



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
