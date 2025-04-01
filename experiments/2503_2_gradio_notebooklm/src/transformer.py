from psiking.core.base.schema import TextNode, Document
from psiking.core.processor.document.text_merger import TextNodeMerger
from psiking.core.splitter.text.langchain_text_splitters import LangchainRecursiveCharacterTextSplitter


class TransformerModule:
    def __init__(self, settings):
        self.settings=settings
        self.merger = TextNodeMerger()
        self.splitter = LangchainRecursiveCharacterTextSplitter(
            chunk_size = settings.text_chunk_size,
            chunk_overlap = settings.text_chunk_overlap
        )
        self.min_text_length = 30
    
    def _merge(self, documents):
        merged_documents = []
        for document in documents:
            merged_document = self.merger.run(document)
            merged_documents.append(merged_document)
        return merged_documents
    
    def _split(self, documents):
        chunks = []
        for document in documents:
            document_chunks = []
            source_id = document.id_
            for i, node in enumerate(document.nodes):
                # Run Splitter
                if isinstance(node, TextNode):
                    try:
                        split_nodes = self.splitter.run(node)
                    except Exception as e:
                        print(i, node)
                        print(str(e))
                        raise e
                else:
                    split_nodes = [node]
                
                # Create New Document
                for split_node in split_nodes:
                    ## Filter TextNodes with short lengths
                    if isinstance(split_node, TextNode) and len(split_node.text.strip())<self.min_text_length:
                        continue
                    
                    # Each Document contains single node
                    chunk = Document(
                        nodes=[split_node],
                        
                        metadata={**document.metadata}
                    )
                    document_chunks.append(chunk)
            chunks.extend(document_chunks)
        return chunks
        
    def run(self, documents):
        documents = self._merge(documents)
        documents = self._split(documents)
        return documents