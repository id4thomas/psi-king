from psiking.core.formatter.node.schema import (
    TextNodeTemplate,
    ImageNodeTemplate,
    TableNodeTemplate,
    TEXT_RESOURCE_KEY,
    IMAGE_RESOURCE_KEY,
    CAPTION_RESOURCE_KEY
)

SimpleTextNodeTemplate = TextNodeTemplate(
    text_template = f"<|{TEXT_RESOURCE_KEY}|>"
)

SimpleImageNodeTextOnlyTemplate = ImageNodeTemplate(
    text_tempalte = f"<|{TEXT_RESOURCE_KEY}|>\nCaption: <|{CAPTION_RESOURCE_KEY}|>"
)

SimpleTableNodeTextOnlyTemplate = TableNodeTemplate(
    text_tempalte = f"<|{TEXT_RESOURCE_KEY}|>\nCaption: <|{CAPTION_RESOURCE_KEY}|>"
)