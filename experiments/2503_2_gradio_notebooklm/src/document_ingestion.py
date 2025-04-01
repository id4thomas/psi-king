from typing import Dict, List

from psiking.core.reader.pdf.docling_reader import DoclingPDFReader

from .reader import ReaderModule
from .transformer import TransformerModule

class DocumentIngestionPipeline:
    def __init__(self, settings):
        self.settings = settings
        
        self.reader = self._load_reader()
        self.transformer = self._load_transformer()
        
    def _load_reader(self):
        return ReaderModule(self.settings)
    
    def _load_transformer(self):
        return TransformerModule(self.settings)
    
    def run(self, file_paths: List[str], source_id_prefix="", extra_infos: Dict[str,str]={}):
        documents = self.reader.run(
            file_paths,
            source_id_prefix=source_id_prefix,
            extra_infos=extra_infos
        )
        
        documents = self.transformer.run(documents)
        return documents