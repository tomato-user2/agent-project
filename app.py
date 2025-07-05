import gradio as gr
from agents import build_graph

graph = build_graph()

async def run_book_recommender(user_input):
    # 1) Kick off the state‐graph via .astream
    initial_state = {"user_input": user_input}
    final_state = None

    try:
        async for state in graph.astream(initial_state):
            final_state = state
    except Exception as e:
        # Log it somewhere if you want
        print("🔥 Exception while streaming graph:", e)
        # And then re-raise so Gradio can show it
        raise

    # 2) If for some bizarre reason the graph yielded zero times,
    #    fall back to a safe default
    if final_state is None:
        final_state = {
            "final_recommendations": [],
            "final_reasoning": "⚠️ Graph never yielded a final state."
        }

    # 3) Extract the real outputs
    recs = final_state.get("final_recommendations", [])
    reasoning = final_state.get("final_reasoning", "")

    # 4) Format them
    recs_text = "\n\n".join(
        f"📘 {r['title']}\n🔗 {r.get('link','')}\n💡 {r.get('reason','')}"
        for r in recs
    ) or "No recommendations found."

    # 5) **Explicitly return** in all cases
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
