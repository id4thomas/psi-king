import json
from typing import Any, Dict, List, Optional

from psiking.core.base.schema import BaseNode, Document


def _validate_is_flat_dict(metadata_dict: dict) -> None:
    """
    Validate that metadata dict is flat,
    and key is str, and value is one of (str, int, float, None).
    """
    for key, val in metadata_dict.items():
        if not isinstance(key, str):
            raise ValueError("Metadata key must be str!")
        if not isinstance(val, (str, int, float, type(None))):
            raise ValueError(
                f"Value for metadata {key} must be one of (str, int, float, None)"
            )

def document_to_metadata_dict(
    document: Document,
    keys: Optional[List[str]] = None,
    flat_metadata: bool = False,
) -> Dict[str, Any]:
    """Common logic for converting document.metadata into dict."""
    if keys is None:
        keys = list(document.metadata.keys())

    metadata: Dict[str, Any] = document.metadata
    metadata = {k:v for k,v in metadata.items() if k in keys}

    if flat_metadata:
        _validate_is_flat_dict(metadata)
        
    return metadata