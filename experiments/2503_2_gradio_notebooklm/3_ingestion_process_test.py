import concurrent.futures as futures
from datetime import datetime
import json
import os
import time
import traceback
from typing import Dict, List
import uuid

from pydantic import BaseModel, Field
from psiking.core.base.schema import doc_to_json

from config import experiment_settings, app_settings
from src.document_ingestion import DocumentIngestionPipeline

os.environ["DOCLING_ARTIFACTS_PATH"] = os.path.join(
    experiment_settings.docling_model_weight_dir, "docling-models"
)

class PipelineSettings(BaseModel):
    # PictureDescription
    vlm_base_url: str = Field("")
    vlm_api_key: str = Field("")
    vlm_model: str = Field("")
    
    # PDF2Img
    poppler_path: str = Field("/opt/homebrew/Cellar/poppler/25.01.0/bin")
    
    # Chunking
    text_chunk_size: int = Field(1024)
    text_chunk_overlap: int = Field(128)
    
    output_dir: str = Field("")

class InputFile(BaseModel):
    uid: str
    file_path: str
    # metadata: Dict[str, str] = Field(dict())

def ingest(
    pipeline_settings: dict,
    input_files: List[dict],
    metadata_dicts: Dict[str, dict]
):
    pipeline_settings = PipelineSettings(**pipeline_settings)
    input_files = [InputFile(**x) for x in input_files]
    
    start = time.time()
    pipeline = DocumentIngestionPipeline(pipeline_settings)
    print("Pipeline Loaded in {:.3f}".format(time.time()-start))
    
    start = time.time()
    documents = pipeline.run(
        [x.file_path for x in input_files],
        source_id_prefix="kr-fsc_policy-pdf",
        extra_infos=metadata_dicts
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
            
    # for input_file, document in zip(input_files, documents):
    #     with open(os.path.join(pipeline_settings.output_dir, f"{input_file.uid}.json"), "w") as f:
    #         f.write(json.dumps(doc_to_json(document), ensure_ascii=False, indent=4))
    
def main():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    collection_key = str(uuid.uuid4())
    print(f"Collection: {collection_key}")
    
    # Initialize output_dir
    output_dir = os.path.join("storage", collection_key)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize Settings
    max_workers = 4
    pipeline_settings = PipelineSettings(
        vlm_base_url = experiment_settings.vlm_base_url,
        vlm_api_key=experiment_settings.vlm_api_key,
        vlm_model = experiment_settings.vlm_model,
        poppler_path="/opt/homebrew/Cellar/poppler/25.01.0/bin",
        text_chunk_size=1024,
        text_chunk_overlap=128,
        output_dir=output_dir
    )
    
    # Load Files
    ## get fnames
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
            file_path=str(os.path.join(file_dir, fname))
        )
        input_files.append(input_file.model_dump())
    with open(f"storage/request-{timestamp}.json", "w") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "input_files": input_files
                },
                indent=4,
                ensure_ascii=False
            ),
        )
        
    print("STARTING INGESTION {}".format(len(input_files)))

    start_tm = time.time()  # 시작 시간
    # ProcessPoolExecutor
    num_workers = min(max_workers, len(input_files))
    batch_size = int(len(input_files)/num_workers)
    
    future_list = []
    with futures.ProcessPoolExecutor(max_workers=num_workers) as excutor:
        for i in range(0, len(input_files), batch_size):
            batch = input_files[i:i+batch_size]
            # map -> 작업 순서 유지, 즉시 실행
            future = excutor.submit(
                ingest,
                pipeline_settings.model_dump(),
                batch,
                metadata_dicts
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
    
if __name__ == "__main__":
    main()