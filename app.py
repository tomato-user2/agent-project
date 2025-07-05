import gradio as gr
from agents import build_graph
import json
from pprint import pformat

graph = build_graph()

async def run_book_recommender(user_input):
    initial_state = {"user_input": user_input}
    final_state = None

    try:
        async for state in graph.astream(initial_state):
            final_state = state
    except Exception as e:
        print("🔥 Exception while streaming graph:", e)
        raise

    if final_state is None:
        final_state = {
            "final_recommendations": [],
            "final_reasoning": "⚠️ Graph never yielded a final state."
        }

    reasoning_data = final_state.get("reasoning", {})
    recs = reasoning_data.get("final_recommendations", [])
    reasoning = reasoning_data.get("final_reasoning", "")

    # Defensive formatting of recommendations
    try:
        if isinstance(recs, list) and all(isinstance(r, dict) for r in recs):
            # Format nicely as before
            recs_text = "\n\n".join(
                f"📘 {r.get('title', 'Unknown Title')}\n🔗 {r.get('link','')}\n💡 {r.get('reason','')}"
                for r in recs
            )
            if not recs_text.strip():
                recs_text = "No recommendations found."
        else:
            # For any other structure, try pretty-printing JSON or just string conversion
            try:
                recs_text = json.dumps(recs, indent=2, ensure_ascii=False)
            except Exception:
                recs_text = pformat(recs)
            if not recs_text.strip():
                recs_text = "No recommendations found."
    except Exception as e:
        recs_text = f"Error formatting recommendations: {e}"

    return recs_text, reasoning

with gr.Blocks() as demo:
    gr.Markdown("# 📚 AI Book Recommender")
    user_in = gr.Textbox(label="Tell me some books you like")
    btn = gr.Button("Get Recommendations")
    out_recs = gr.Textbox(label="Recommended Books", lines=10)
    out_reason = gr.Textbox(label="Reasoning / Debug Log", lines=15)

    btn.click(
      fn=run_book_recommender,
      inputs=user_in,
      outputs=[out_recs, out_reason],
    )

if __name__=="__main__":
    demo.launch()
