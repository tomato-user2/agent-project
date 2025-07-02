import gradio as gr
from agent import agent

def chat_with_agent(user_input, chat_history):
    response = agent.run(user_input)
    chat_history.append((user_input, response))
    return chat_history, chat_history

with gr.Blocks() as demo:
    gr.Markdown("## 📚 Book Recommendation Agent (powered by LLaMA3 + smolagents)")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Tell me a few books you like...", label="Your favorite books")
    clear = gr.Button("Clear")

    msg.submit(chat_with_agent, [msg, chatbot], [chatbot, chatbot])
    clear.click(lambda: ([], ""), None, [chatbot, msg])

demo.launch()
