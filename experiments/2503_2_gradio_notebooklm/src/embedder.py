import os
import sys
from typing import List

from fastembed import SparseTextEmbedding
from visual_bge.modeling import Visualized_BGE
from pydantic import BaseModel, Field


sys.path.append("/Users/id4thomas/github/psi-king/src/psiking")
from core.base.schema import TextNode, ImageNode, TableNode, Document
from core.embedder.flagembedding import (
    VisualizedBGEInput, 
    LocalVisualizedBGEEmbedder
)
from core.embedder.fastembed.local_sparse import LocalFastEmbedSparseEmbedder

from core.formatter.document.simple import SimpleTextOnlyFormatter

class SparseEmbeddingOutput(BaseModel):
    values: List[List[float]]
    indices: List[List[int]]

class DenseEmbeddingOutput(BaseModel):
    values: List[List[float]]
    
class EmbeddingOutput(BaseModel):
    texts: List[str]
    dense: DenseEmbeddingOutput
    sparse: SparseEmbeddingOutput


class EmbedderModule:
    def __init__(self, settings):
        self.settings = settings
        
        self.text_formatter = self._load_text_formatter()
        
        self.dense_embedder = self._load_dense_embedder()
        print("Loaded Dense Embedder")
        self.sparse_embedder = self._load_sparse_embedder()
        print("Loaded Sparse Embedder")

    def _load_text_formatter(self):
        return SimpleTextOnlyFormatter()
    
    # BGE-Visualized
    def _load_dense_embedder(self):
        bge_m3_model_dir = os.path.join(
            self.settings.embedding_model_path, "bge-m3"
        )
        visualized_model_dir = os.path.join(
            self.settings.embedding_model_path, "baai-bge-visualized/Visualized_m3.pth"
        )

        dense_embedding_model = Visualized_BGE(
            model_name_bge = bge_m3_model_dir,
            model_weight= visualized_model_dir
        )
        dense_embedding_model.eval()
        dense_embedding_model.dtype
        dense_embedder = LocalVisualizedBGEEmbedder(
            model=dense_embedding_model
        )
        return dense_embedder
    
    def _format_texts(self, document: Document) -> str:
        formatted_text = self.text_formatter.run([document])[0]
        return formatted_text
    
    def _prepare_visualized_bge_input(self, document: Document):
        # Single 
        formatted_text = self._format_texts(document)
        
        node = document.nodes[0]
        if isinstance(node, TextNode):
            return VisualizedBGEInput(text=formatted_text)
        elif isinstance(node, ImageNode) or isinstance(node, TableNode):
            return VisualizedBGEInput(
                text=formatted_text,
                image=node.image
            )
        else:
            raise ValueError("Unknown node type error {}".format(type(node)))
        
    def _load_sparse_embedder(self):
        sparse_model = SparseTextEmbedding(
            model_name="Qdrant/bm42-all-minilm-l6-v2-attentions",
            specific_model_path=os.path.join(
                self.settings.embedding_model_path, "Qdrant/all_miniLM_L6_v2_with_attentions"
            ),
            cuda=False,
            lazy_load=False
        )
        sparse_embedder = LocalFastEmbedSparseEmbedder(
            model=sparse_model
        )
        return sparse_embedder
    
    def _dense_embed(self, documents: List[Document]) -> DenseEmbeddingOutput:
        visualized_bge_inputs = [
            self._prepare_visualized_bge_input(x) for x in documents
        ]
        dense_embeddings = self.dense_embedder.run(
            visualized_bge_inputs,
            batch_size = self.settings.dense_batch_size,
            disable_tqdm=False
        )
        return DenseEmbeddingOutput(values=dense_embeddings)
    
    def _sparse_embed(self, documents: List[Document]) -> SparseEmbeddingOutput:
        sparse_inputs = [
            self._format_texts(x) for x in documents
        ]
        values, indices = self.sparse_embedder.run(
            sparse_inputs,
            batch_size=self.settings.sparse_batch_size
        )
        return SparseEmbeddingOutput(
            values=values,
            indices=indices
        )

    def run(self, documents):
        formatted_texts  = [
            self._format_texts(x) for x in documents
        ]
        
        dense_embedding_out = self._dense_embed(documents)
        sparse_embedding_out = self._sparse_embed(documents)
        return EmbeddingOutput(
            texts=formatted_texts,
            dense=dense_embedding_out,
            sparse=sparse_embedding_out
        )