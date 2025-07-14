# allganize-rag-evaluation Multimodal Hybrid Indexing (Updated 2025.07)
## 1. Methodology
### 1-1. Chat Template
Document (Passage):
```
[
    {
        'role': 'user',
        'content': [
                {'type': 'text', 'text': text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                }
            ]
        }
    ]
```

Query:
`Query: ...`


## 2. Performance
### 2-1. File Level
**mean Average Precision**
| method | @5 | @10 | @15 |
| --- | --- | --- | --- |
| dense | 0.7662 | 0.7230 | 0.6810 |
| hybrid | 0.6094 | 0.5752 | 0.5455 |

**mean Reciprocal Rank**
| method | @5 | @10 | @15 |
| --- | --- | --- | --- |
| dense | 0.8594 | 0.8594 | 0.8594 |
| hybrid | 0.7653 | 0.7697 | 0.7697 |



### 2-2. File+Page Level
**mean Average Precision**
| method | @5 | @10 | @15 |
| --- | --- | --- | --- |
| dense | 0.2208 | 0.1648 | 0.1349 |
| hybrid | 0.1604 | 0.1279 | 0.1080 |

**mean Reciprocal Rank**
| method | @5 | @10 | @15 |
| --- | --- | --- | --- |
| dense | 0.4428 | 0.4546 | 0.4569 |
| hybrid | 0.3172 | 0.3320 | 0.3370 |