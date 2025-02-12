from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union, TYPE_CHECKING

from PIL import Image
import torch
from tqdm import tqdm

from core.embedder.base import BaseEmbedder

if TYPE_CHECKING:
    from visual_bge.modeling import Visualized_BGE

@dataclass
class VisualizedBGEInput:
    text: str = ""
    image: Optional[Image.Image] = None

class LocalVisualizedBGEEmbedder(BaseEmbedder):
    """Embedder using visual_bge.Visualized_BGE locally"""
    
    def __init__(
        self,
        model: "Visualized_BGE",
        text_tokenizer_args: Optional[dict] = None
    ):
        """
        model should be loaded & injected from outside
        ```
        # https://github.com/FlagOpen/FlagEmbedding/blob/b2871ac56146856c1b3a2688d04d77868c461f67/research/visual_bge/visual_bge/modeling.py#L43
        # term 'bge-m3' must be in `model_name_bge`
        bge_m3_model_dir = ".../bge-m3"
        visualized_model_dir=".../baai-bge-visualized/Visualized_m3.pth"
        Visualized_BGE(
            model_name_bge = bge_m3_model_dir,
            model_weight= visualized_model_dir
        )
        ```
        """
        self.model=model
        if not text_tokenizer_args is None:
            self.text_tokenizer_args = text_tokenizer_args
        else:
            self.text_tokenizer_args = {"padding": True}
    
    def prepare_text_inputs(self, inputs: List[VisualizedBGEInput]):
        text_inputs = self.model.tokenizer(
            [x.text for x in inputs],
            return_tensors="pt",
            **self.text_tokenizer_args
        )
        return text_inputs
    
    def transform_image(self, image: Image.Image) -> "torch.tensor":
        return self.model.preprocess_val(image).unsqueeze(0)
    
    def prepare_image_inputs(self, inputs: List[VisualizedBGEInput]):
        image_inputs = []
        for input in inputs:
            image_inputs.append(self.transform_image(input.image))
        image_inputs = torch.cat(image_inputs, dim=0)
        return image_inputs
    
    @torch.no_grad()
    def encode_text(
        self,
        inputs: List[VisualizedBGEInput],
        batch_size: int=16,
        disable_tqdm: bool = True
    ) -> List[List[float]]:
        if len(inputs)==0:
            return []
        outputs = []
        for i in tqdm(range(0, len(inputs), batch_size), disable=disable_tqdm):
            batch_inputs = self.prepare_text_inputs(inputs[i:i+batch_size])
            batch_output = self.model.encode_text(
                batch_inputs.to(self.model.device)
            ).cpu().detach()
            outputs.append(batch_output)
        return torch.cat(outputs, dim=0).tolist()
        
    @torch.no_grad()
    def encode_mm(
        self,
        inputs: List[VisualizedBGEInput],
        batch_size: int=16,
        disable_tqdm: bool = True
    ) -> List[List[float]]:
        if len(inputs)==0:
            return []
        outputs = []
        for i in tqdm(range(0, len(inputs), batch_size), disable=disable_tqdm):
            batch_text_inputs = self.prepare_text_inputs(inputs[i:i+batch_size])
            batch_image_inputs = self.prepare_image_inputs(inputs[i:i+batch_size])
            batch_output = self.model.encode_mm(
                images=batch_image_inputs.to(self.model.dtype).to(self.model.device),
                texts=batch_text_inputs.to(self.model.device)
            ).cpu().detach()
            outputs.append(batch_output)
        return torch.cat(outputs, dim=0).tolist()

    def encode(self, image: Image.Image=None, text: str=None) -> Optional[torch.tensor]:
        """For Simple inference (ex. query)"""
        if image is not None:
            image = self.model.preprocess_val(image).unsqueeze(0)
            if text is not None:
                text = self.model.tokenizer(text, return_tensors="pt", padding=True)
                return self.model.encode_mm(
                    image.to(self.model.dtype).to(self.model.device),
                    text.to(self.model.device)
                ).cpu().detach()
            else:
                return model.encode_image(
                    image.to(model.dtype).to(model.device)
                ).cpu().detach()
        else:
            if text is not None:
                text = self.model.tokenizer(text, return_tensors="pt", padding=True)
                return self.model.encode_text(
                    text.to(self.model.device)
                ).cpu().detach()
            else:
                return None

    def run(
        self,
        inputs: List[VisualizedBGEInput],
        batch_size:int=16,
        disable_tqdm: bool = True
    ) -> List[List[float]]:
        text_only_idxs = [i for i in range(len(inputs)) if inputs[i].image is None]
        
        ## Process Embed
        mm_embeds = self.encode_mm(
            [x for i, x in enumerate(inputs) if i not in text_only_idxs],
            batch_size=batch_size,
            disable_tqdm=disable_tqdm
        )
        text_embeds = self.encode_text(
            [x for i, x in enumerate(inputs) if i in text_only_idxs],
            batch_size=batch_size,
            disable_tqdm=disable_tqdm
        )
        
        ## Aggregate & Return
        outputs = []
        mm_i = 0
        text_i = 0
        for i in range(len(inputs)):
            if i in text_only_idxs:
                outputs.append(text_embeds[text_i])
                text_i +=1
            else:
                outputs.append(mm_embeds[mm_i])
                mm_i +=1
        return outputs