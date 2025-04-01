
import concurrent.futures as futures
import base64
from datetime import datetime
from io import BytesIO
import json
import os
from PIL import Image
import time
from typing import Dict, List
import traceback
import uuid

import gradio as gr
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models

from psiking.core.base.schema import TextNode, ImageNode, TableNode, Document
from psiking.core.base.schema import doc_to_json, json_to_doc
from psiking.core.storage.vectorstore.qdrant import QdrantSingleHybridVectorStore

from config import experiment_settings, app_settings

from src.document_ingestion import DocumentIngestionPipeline
from src.embedder import EmbedderModule
from src.models import PipelineSettings, EmbeddingSettings, InputFile

os.environ["DOCLING_ARTIFACTS_PATH"] = os.path.join(
    experiment_settings.docling_model_weight_dir, "docling-models"
)

PIPELINE_SETTINGS = PipelineSettings(
    vlm_base_url = experiment_settings.vlm_base_url,
    vlm_api_key=experiment_settings.vlm_api_key,
    vlm_model = experiment_settings.vlm_model,
    poppler_path="/opt/homebrew/Cellar/poppler/25.01.0/bin",
    text_chunk_size=1024,
    text_chunk_overlap=128,
    output_dir=""
)

EMBEDDING_SETTINGS = EmbeddingSettings(
    embedding_model_path=str(
        os.path.join(experiment_settings.model_weight_dir, "embedding")
    ),
    dense_batch_size=4,
    sparse_batch_size=256
)
embedder = EmbedderModule(EMBEDDING_SETTINGS)
qdrant_client = QdrantClient(host="localhost", port=app_settings.qdrant_port)

# Document Ingestion
## Ingestion
def ingest_fn(
    pipeline_settings: dict,
    input_files: List[dict]
):
    pipeline_settings = PipelineSettings(**pipeline_settings)
    input_files = [InputFile(**x) for x in input_files]
    
    start = time.time()
    pipeline = DocumentIngestionPipeline(pipeline_settings)
    print("Pipeline Loaded in {:.3f}".format(time.time()-start))
    
    start = time.time()
    documents = pipeline.run(
        input_files,
        source_id_prefix="kr-fsc_policy-pdf"
    )
    print("Pipeline Finished in {:.3f}".format(time.time()-start))
    
    ## Save
    for input_file in input_files:
        fname = input_file.file_path.rsplit("/",1)[-1]
        chunks = list(filter(lambda x: x.metadata['source_file']==fname, documents))
        with open(os.path.join(pipeline_settings.output_dir, f"{input_file.uid}.json"), "w") as f:
            f.write(json.dumps(
                {
                    "file_path": input_file.file_path,
                    "chunks": [
                        doc_to_json(x) for x in chunks
                    ]
                },
                ensure_ascii=False,
                indent=4
        ))
            
def ingest(collection_key, file_paths, max_workers=4):
    output_dir = os.path.join("storage", collection_key)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    PIPELINE_SETTINGS.output_dir=output_dir
    
    input_files = []
    for file_path in file_paths:
        input_file = InputFile(
            file_path=file_path
        )
        input_files.append(input_file)
        
    start_tm = time.time()  # 시작 시간
    # ProcessPoolExecutor
    num_workers = min(max_workers, len(input_files))
    batch_size = int(len(input_files)/num_workers)
    
    future_list = []
    with futures.ProcessPoolExecutor(max_workers=num_workers) as excutor:
        for i in range(0, len(input_files), batch_size):
            batch = [x.model_dump() for x in input_files[i:i+batch_size]]
            # map -> 작업 순서 유지, 즉시 실행
            future = excutor.submit(
                ingest_fn,
                PIPELINE_SETTINGS.model_dump(),
                batch
            )
            future_list.append(future)
            
    results = []
    for future in futures.as_completed(future_list):
        try:
            results.append(future.result())
        except Exception as e:
            print(f"Error processing batch: {e}")
            traceback.print_exc()

    end_tm = time.time() - start_tm  # 종료 시간
    msg = '\n Result -> {} Time : {:.2f}s'  # 출력 포맷
    print(msg.format(list(input_files), end_tm))  # 최종 결과
    return input_files
        
## Embed
def embed(collection_key, input_files):
    chunks = []
    for input_file in input_files:
        with open(f"storage/{collection_key}/{input_file.uid}.json", "r") as f:
            document_data = json.load(f)
        document_chunks = [
            json_to_doc(x) for x in document_data["chunks"]
        ]
        chunks.extend(document_chunks)
    
    embeddings = embedder.run(chunks)
    
    # Insert
    
    vector_store = QdrantSingleHybridVectorStore(
        collection_name=collection_key,
        client=qdrant_client
    )
    dense_embedding_dim=1024
    dense_vectors_config = models.VectorParams(
        size=dense_embedding_dim,
        distance=models.Distance.COSINE,
        on_disk=True,
        hnsw_config = {
            "m": 16,
            "ef_construct": 100,
        }
    )

    # Sparse BM42 Embedding
    sparse_vectors_config = models.SparseVectorParams(
        modifier=models.Modifier.IDF, ## uses indices from bm42 embedder
    )

    # Create VectorStore
    vector_store.create_collection(
        dense_vector_config=dense_vectors_config,
        sparse_vector_config=sparse_vectors_config,
        on_disk_payload=True,
    )
    vector_store.add(
        documents=chunks,
        texts=embeddings.texts,
        dense_embeddings=embeddings.dense.values,
        sparse_embedding_values=embeddings.sparse.values,
        sparse_embedding_indices=embeddings.sparse.indices,
        metadata_keys=["source_file", "title"]
    )

def process(collection_key, files):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    
    start = time.time()
    try:
        input_files=ingest(collection_key, files)
        yield "Indexing {} files complete in {:.3f}.".format(
            len(files),
            time.time()-start
        )
    except Exception as e:
        traceback.print_exc()
        yield f"Indexing failed: {str(e)}"
    
    with open(f"storage/collections/{collection_key}.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "input_files": [x.model_dump() for x in input_files]
                },
                indent=4,
                ensure_ascii=False
            ),
        )
        
    try:
        embed(collection_key, input_files)
    except Exception as e:
        traceback.print_exc()
        yield f"Embedding failed: {str(e)}"
        
    yield "Embedding {} files complete in {:.3f}.".format(
            len(files),
            time.time()-start
        )

def search(collection_key, query):
    vector_store = QdrantSingleHybridVectorStore(
        collection_name=collection_key,
        client=qdrant_client
    )
    
    query_document = Document(
        nodes=[TextNode(text=query)]
    )
    query_embedding_output = embedder.run([query_document])
    results = vector_store.query(
        mode="hybrid",
        dense_embedding=query_embedding_output.dense.values[0],
        sparse_embedding_values=query_embedding_output.sparse.values[0],
        sparse_embedding_indices=query_embedding_output.sparse.indices[0],
        limit=10
    )
    print(results)

def load_image(file_path):
    return Image.open(file_path)

def echo(message, history):
    print(history)
    
    img = load_image("./resources/test.jpeg")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    md = f"""
### Here is an image loaded with PIL and embedded in Markdown:
<img src="data:image/png;base64,{img_base64}" width="300">
"""
    msg = gr.ChatMessage(
        # content=gr.Image()
        content = gr.Markdown(md)
        # content=[
        #     gr.Image("/file=resources/test.png"),
        #     gr.Markdown("some response")
        # ]
    )
    return msg
    # return message

def main():
    with gr.Blocks(fill_height=True) as demo:
        gr.Markdown("Document Chat")

        with gr.Row():    
            with gr.Column(scale=2):
                collection_key = gr.Textbox(
                    value=str(uuid.uuid4()),
                    label="Collection:",
                    placeholder="Enter collection key here..."
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

            with gr.Column(scale=8):
                # chatbot = gr.Chatbot(
                #     label="Chatbot",
                #     type="messages",
                #     show_copy_button=True
                # )
                chat = gr.ChatInterface(
                    fn=echo,
                    type="messages",
                    examples=["hello", "hola", "merhaba"],
                    title="Chat Bot"
                )
        index_button.click(
            process, 
            inputs=[collection_key, file_upload], 
            outputs=indexing_status
        )

        demo.launch()
    
if __name__ == "__main__":
    main()