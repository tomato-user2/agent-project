import gradio as gr
from agents import build_graph
import asyncio

# Build the LangGraph once
graph = build_graph()

async def run_book_recommender(user_input):
    initial_state = {"user_input": user_input}

    # Consume the generator until completion
    async for state in graph.astream(initial_state):
        final_state = state  # This keeps updating with each step

    recommendations = final_state.get("recommendations", [])
    reasoning = final_state.get("reasoning", "")

    recommendations_text = "\n\n".join(
        [f"📘 {rec['title']}\n🔗 {rec['link']}\n📝 {rec['snippet']}" for rec in recommendations]
    ) or "No recommendations found."

    return recommendations_text, reasoning

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📚 AI Book Recommender")
    user_input = gr.Textbox(label="Tell me some books you like")
    recommend_btn = gr.Button("Get Recommendations")
    recommendations_output = gr.Textbox(label="Recommended Books")
    reasoning_output = gr.Textbox(label="Reasoning Steps")

    recommend_btn.click(run_book_recommender, inputs=user_input, outputs=[recommendations_output, reasoning_output])

if __name__ == "__main__":
    demo.launch()
