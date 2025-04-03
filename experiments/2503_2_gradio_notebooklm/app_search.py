import base64
import concurrent.futures as futures
from datetime import datetime
from io import BytesIO
import json
import os
import time
from typing import Dict, List
from tqdm import tqdm
import traceback
import uuid

import gradio as gr
import pandas as pd
from PIL import Image
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

# Initialization
def initialize_collection(collection_key):
    collection_dir = os.path.join("storage/collections", collection_key)
    if not os.path.exists(collection_dir):
        os.makedirs(collection_dir)
        
    document_dir = os.path.join(collection_dir, "documents")
    if not os.path.exists(document_dir):
        os.makedirs(document_dir)
        
    metadata_dir = os.path.join(collection_dir, "metadata")
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

def prepare_input_files(collection_key):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    file_dir = os.path.join(
        experiment_settings.data_dir, "retrieval_dataset/2503-01-korean-finance/kr-fsc_policy"
    )
    fnames = [x for x in os.listdir(file_dir) if "pdf" in x][:2]
    ## get metadata
    metadata_file_path =  os.path.join(
        experiment_settings.data_dir, "retrieval_dataset/2503-01-korean-finance/kr-fsc_pdf_file_metadata.json"
    )
    with open(metadata_file_path, "r") as f:
        metadata_file_contents = json.load(f)
    metadata_dicts = {
        "{}_{}.pdf".format(x["item_id"], x["no"]): {
            "title": x["item_title"]
        }
        for x in metadata_file_contents
    }
    ## prepare input files
    input_files = []
    for fname in fnames:
        input_file = InputFile(
            uid=str(uuid.uuid4()),
            file_path=str(os.path.join(file_dir, fname)),
            metadata=metadata_dicts[fname]
        )
        input_files.append(input_file)

    with open(f"storage/request-{timestamp}.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "collection_key": collection_key,
                    "timestamp": timestamp,
                    "input_file_ids": [x.uid for x in input_files]
                },
                indent=4,
                ensure_ascii=False
            ),
        )
    return input_files

############# INDEX
## 1. Ingestion
def ingest_fn(
    pipeline_settings: dict,
    collection_key: str,
    input_files: List[dict]
):
    pipeline_settings = PipelineSettings(**pipeline_settings)
    collection_dir = os.path.join("storage/collections", collection_key)
    input_files = [InputFile(**x) for x in input_files]
    
    start = time.time()
    pipeline = DocumentIngestionPipeline(pipeline_settings)
    print("Pipeline Loaded in {:.3f}".format(time.time()-start))
    
    start = time.time()
    documents = pipeline.run(input_files)
    print("Pipeline Finished in {:.3f}".format(time.time()-start))
    
    ## Save
    for input_file in input_files:
        fname = input_file.file_path.rsplit("/",1)[-1]
        chunks = list(filter(lambda x: x.metadata['source_file']==fname, documents))
        
        # Save Documents
        chunk_ids = []
        for chunk in chunks:
            chunk_id = chunk.id_
            chunk_ids.append(chunk_id)
            with open(os.path.join(collection_dir, f"documents/{chunk_id}.json"), "w") as f:
                f.write(json.dumps(doc_to_json(chunk), indent=4, ensure_ascii=False))
        
        # Save Metadata
        input_file_metadata = {
            "uid": input_file.uid,
            "file_path": input_file.file_path,
            "chunk_ids": chunk_ids
        }
        with open(os.path.join(collection_dir, f"metadata/{input_file.uid}.json"), "w") as f:
            f.write(json.dumps(input_file_metadata, indent=4, ensure_ascii=False))
            
def ingest(collection_key, input_files, max_workers=4):
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
                collection_key,
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
def embed(collection_key, input_files, batch_size = 32):
    # Initialize Collection
    collection_dir = os.path.join("storage/collections", collection_key)
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
    
    # Get Chunk Ids
    chunk_ids = []
    for input_file in input_files:
        with open(os.path.join(collection_dir, f"metadata/{input_file.uid}.json"), "r") as f:
            input_file_metadata = json.load(f)
            input_file_chunk_ids = input_file_metadata["chunk_ids"]
        chunk_ids.extend(input_file_chunk_ids)

    # Batched Embedding
    for i in tqdm(range(0, len(chunk_ids), batch_size)):
        batch_chunk_ids = chunk_ids[i:i+batch_size]
        batch = []
        for chunk_id in batch_chunk_ids:
            with open(os.path.join(collection_dir, f"documents/{chunk_id}.json"), "r") as f:
                chunk = json_to_doc(json.load(f))
            batch.append(chunk)
        
        embeddings = embedder.run(batch)
        vector_store.add(
            documents=batch,
            texts=embeddings.texts,
            dense_embeddings=embeddings.dense.values,
            sparse_embedding_values=embeddings.sparse.values,
            sparse_embedding_indices=embeddings.sparse.indices,
            metadata_keys=["source_id", "source_file"]
        )

def index(collection_key, files):
    # 1. Initialize
    initialize_collection(collection_key)
    input_files = prepare_input_files(collection_key)
    
    start = time.time()
    # 2. Ingest
    try:
        ingest(collection_key, input_files)
        yield "Indexing {} files complete in {:.3f}.".format(
            len(files),
            time.time()-start
        )
    except Exception as e:
        traceback.print_exc()
        yield f"Indexing failed: {str(e)}"
    
    # 3. Embed
    try:
        embed(collection_key, input_files)
    except Exception as e:
        traceback.print_exc()
        yield f"Embedding failed: {str(e)}"
        
    yield "Embedding {} files complete in {:.3f}.".format(
            len(files),
            time.time()-start
        )

############# SEARCH
def get_document(collection_key, document_id):
    with open(f"storage/collections/{collection_key}/documents/{document_id}.json", "r") as f:
        document_data = json.load(f)
    return json_to_doc(document_data)

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def resize_image(image, ratio=0.3):
    width, height = image.size
    new_size = (int(width * ratio), int(height * ratio))
    image.thumbnail(new_size)
    return image
    
def search(collection_key, query, mode="hybrid"):
    vector_store = QdrantSingleHybridVectorStore(
        collection_name=collection_key,
        client=qdrant_client
    )
    
    query_document = Document(
        nodes=[TextNode(text=query)]
    )
    query_embedding_output = embedder.run([query_document])
    if mode=="hybrid":
        results = vector_store.query(
            mode="hybrid",
            dense_embedding=query_embedding_output.dense.values[0],
            sparse_embedding_values=query_embedding_output.sparse.values[0],
            sparse_embedding_indices=query_embedding_output.sparse.indices[0],
            limit=10
        )
    elif mode=="dense":
        results = vector_store.query(
            mode="hybrid",
            dense_embedding=query_embedding_output.dense.values[0],
            limit=10
        )
    else:
        raise ValueError("search mode should be dense or hybrid")
    
    result = {
        "score": [],
        "source": [],
        "id": [],
        "text": [],
        "image": []
    }
    for point in results.points:
        point_id = point.id
        document = get_document(collection_key, point_id)
        node = document.nodes[0]
        if isinstance(node, TextNode):
            text = node.text[:100]
            img = ""
        elif isinstance(node, ImageNode):
            text = node.caption[:100]
            img = node.image
            img = resize_image(img, ratio=0.5)
            img_base64 = image_to_base64(img)
            img = f"![img](data:image/png;base64,{img_base64})"
        elif isinstance(node, TableNode):
            text = node.text[:100]
            img = node.image
            img = resize_image(img, ratio=0.5)
            img_base64 = image_to_base64(img)
            img = f"![img](data:image/png;base64,{img_base64})"
        else:
            continue
        
        result["score"].append(point.score)
        result["source"].append(document.metadata['source_file'])
        result["id"].append(document.id_)
        result["text"].append(text)
        result["image"].append(img)
    df = pd.DataFrame(result)
    return df

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
                # chat = gr.ChatInterface(
                #     fn=echo,
                #     type="messages",
                #     examples=["hello", "hola", "merhaba"],
                #     title="Chat Bot"
                # )
                query_input = gr.Textbox(label="Query", lines=3)
                result_df = gr.DataFrame(
                    # datatype=["number", "text", "text", "text"],
                    # column_widths=["5%", "7%", "8%", "60%"],
                    datatype=["number", "text", "text", "text", "markdown"],
                    column_widths=["8%", "10%", "7%", "45%", "30%"],
                    wrap=True
                )
                search_button = gr.Button("Search")
                
        index_button.click(
            index, 
            inputs=[collection_key, file_upload], 
            outputs=indexing_status
        )
        search_button.click(
            search,
            inputs=[collection_key, query_input],
            outputs=result_df
        )

        demo.launch()
    
if __name__ == "__main__":
    main()