# agent.py
from tools import extract_books, recommend_similar_books

class SimpleBookAgent:
    def run(self, user_input):
        books = extract_books(user_input)
        recommendations = recommend_similar_books(books)

        return {
            "search_snippets": recommendations.get("search_snippets", ""),
            "llm_prompt": recommendations.get("llm_prompt", ""),
            "llm_response": recommendations.get("llm_response", ""),
            "recommendations": recommendations.get("recommendations", []),
        }

agent = SimpleBookAgent()
