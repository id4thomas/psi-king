from typing import Dict, List

from psiking.core.reader.pdf.docling_reader import DoclingPDFReader

from .models import InputFile
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
    
    def run(self, input_files: List[InputFile]):
        documents = self.reader.run(input_files)
        documents = self.transformer.run(documents)
        return documents