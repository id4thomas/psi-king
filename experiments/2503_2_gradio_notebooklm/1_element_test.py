import os
import tempfile
import gradio as gr

def ingest(collection_name: str, files):
    if not files:
        return "No files uploaded."
    return f"Collection '{collection_name}' Ingestion complete"

def chat(query, history, sources):
    response = f"Test Response to query '{query}'"
    history.append((query, response))
    sources.append({"Doc ID": str(len(history)), "Source": "..."})
    return history, history, "", sources

# Define the Gradio interface using Blocks
# with gr.Blocks() as demo:
#     gr.Markdown("## PDF RAG with LlamaIndex and GPT-4")
#     with gr.Row():
#         file_upload = gr.File(label="Upload PDF Files", file_count="multiple", file_types=[".pdf"])
#         index_button = gr.Button("Start Indexing")
#     indexing_status = gr.Textbox(label="Indexing Status", interactive=False)
    
#     gr.Markdown("### Chat with your indexed PDFs")
#     with gr.Row():
#         chat_input = gr.Textbox(label="Enter your query")
#         chat_output = gr.Textbox(label="Chat Response", interactive=False)

#     # Bind the button to the indexing function
#     index_button.click(ingest, inputs=file_upload, outputs=indexing_status)
#     # Bind the chat input submission to the query function
#     # chat_input.submit(chat, inputs=chat_input, outputs=chat_output)
    
#     state = gr.State([])
#     chatbot = gr.Chatbot(label="Chatbot")
#     chat_input.submit(chat, inputs=[chat_input, state], outputs=[chatbot, state, chat_input])

with gr.Blocks() as demo:
    gr.Markdown("## PSI-King Notebook LM")

    with gr.Row():
        # Document Upload & Index
        with gr.Column(scale=2):
            collection_name = gr.Textbox(
                label="Collection:",
                placeholder="Enter collection name here..."
            )
            file_upload = gr.File(
                label="Upload PDF Files",
                file_count="multiple",
                file_types=[".pdf"]
            )
            index_button = gr.Button("Start Indexing")
            indexing_status = gr.Textbox(
                label="Indexing Status",
                interactive=False
            )

        # Right side (8 parts)
        with gr.Column(scale=8):
            # gr.Markdown("### Chat with your indexed PDFs")
            chatbot = gr.Chatbot(label="Chatbot")
            chat_input = gr.Textbox(
                label="Your Message",
                placeholder="Enter your query here..."
            )
            state = gr.State([])
            sources_df = gr.DataFrame(label="Sources Used", headers=["Doc ID", "Source"], datatype=["str", "str"])

    # Bind the button to the indexing function
    index_button.click(
        fn=ingest,
        inputs=[collection_name, file_upload],
        outputs=indexing_status
    )
    # Bind the chat input submission to the query function
    chat_input.submit(
        fn=chat,
        inputs=[chat_input, state, sources_df],
        outputs=[chatbot, state, chat_input, sources_df]
    )

# Launch the Gradio app
demo.launch()