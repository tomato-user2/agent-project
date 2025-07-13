import gradio as gr
from agents import build_graph
import json
from pprint import pformat

graph = build_graph()

async def run_book_recommender(user_input):
    initial_state = {"user_input": user_input}
    final_state = None

    try:
        step_count = 0
        async for state in graph.astream(initial_state):
            step_count += 1
            print(f"🔍 Step {step_count}: State keys = {list(state.keys())}")
            if "final_recommendations" in state:
                print(f"🔍 Step {step_count}: Found final_recommendations: {state['final_recommendations']}")
            if "final_reasoning" in state:
                print(f"🔍 Step {step_count}: Found final_reasoning (first 200 chars): {state['final_reasoning'][:200]}...")
            final_state = state
        print(f"✅ Graph completed in {step_count} steps")
    except Exception as e:
        print("🔥 Exception while streaming graph:", e)
        import traceback
        print("🔥 Traceback:", traceback.format_exc())
        raise

    if final_state is None:
        final_state = {
            "final_recommendations": [],
            "final_reasoning": "⚠️ Graph never yielded a final state."
        }
    
    # Ensure we have the expected keys in final_state
    print(f"🔍 Final state keys: {list(final_state.keys())}")
    print(f"🔍 Final state content: {final_state}")
    
    if "final_recommendations" not in final_state:
        print("⚠️ final_recommendations not found in final state")
        final_state["final_recommendations"] = []
    if "final_reasoning" not in final_state:
        print("⚠️ final_reasoning not found in final state")
        final_state["final_reasoning"] = "⚠️ Missing reasoning data from graph execution."

    # Access the final state - check both possible structures
    recs = final_state.get("final_recommendations", [])
    reasoning = final_state.get("final_reasoning", "")
    
    # If not found in direct keys, check if they're nested under 'reasoning'
    if not recs and "reasoning" in final_state:
        reasoning_data = final_state.get("reasoning", {})
        if isinstance(reasoning_data, dict):
            recs = reasoning_data.get("final_recommendations", [])
            reasoning = reasoning_data.get("final_reasoning", reasoning)
    
    print(f"🔍 Extracted recs: {recs}")
    print(f"🔍 Extracted reasoning (first 200 chars): {reasoning[:200] if reasoning else 'None'}...")

    # Defensive formatting of recommendations
    try:
        # Ensure recs is a list
        if not isinstance(recs, list):
            recs = []
        
        # Filter out invalid entries
        valid_recs = []
        for r in recs:
            if isinstance(r, dict) and r.get('title'):
                valid_recs.append(r)
        
        if valid_recs:
            # Format nicely as before
            recs_text = "\n\n".join(
                f"📘 {r.get('title', 'Unknown Title')}\n🔗 {r.get('link','')}\n💡 {r.get('reason','')}"
                for r in valid_recs
            )
        else:
            recs_text = "No recommendations found."
            
    except Exception as e:
        print(f"Error formatting recommendations: {e}")
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
