# allganize-rag-evaluation Text Hybrid Indexing (Updated 2025.07)
## 1. Methodology
## 1-1. Template
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
| method | query_mode | @5 | @10 | @15 |
| --- | --- | --- | --- | --- |
| sparse | text | 0.3696 | 0.3293 | 0.2985 |
| sparse | messages | 0.3696 | 0.3293 | 0.2985 |
| dense | text | 0.5221 | 0.4891 | 0.4616 |
| dense | messages | 0.5238 | 0.4910 | 0.4616 |
| hybrid | text | 0.4867 | 0.4497 | 0.4216 |
| hybrid | messages | 0.4854 | 0.4484 | 0.4208 |

**mean Reciprocal Rank**
| method | query_mode | @5 | @10 | @15 |
| --- | --- | --- | --- | --- |
| sparse | text | 0.4789 | 0.4840 | 0.4840 |
| sparse | messages | 0.4789 | 0.4840 | 0.4840 |
| dense | text | 0.6097 | 0.6097 | 0.6097 |
| dense | messages | 0.6139 | 0.6139 | 0.6139  |
| hybrid | text | 0.5944 | 0.5944 | 0.5944 |
| hybrid | messages | 0.5931 | 0.5931 | 0.5931 |


### 2-2. File+Page Level
**mean Average Precision**
| method | query_mode | @5 | @10 | @15 |
| --- | --- | --- | --- | --- |
| sparse | text | 0.0301 | 0.0323 | 0.0292 |
| sparse | messages |  0.0301 | 0.0323 | 0.0292 |
| dense | text | 0.1606 | 0.1186 | 0.0978 |
| dense | messages | 0.1627 | 0.1229 | 0.1012 |
| hybrid | text | 0.1147 | 0.0923 | 0.0781 |
| hybrid | messages | 0.1116 | 0.0915 | 0.0787 |


**mean Reciprocal Rank**
| method | query_mode | @5 | @10 | @15 |
| --- | --- | --- | --- | --- |
| sparse | text | 0.0628 | 0.0628 | 0.0642 |
| sparse | messages | 0.0628 | 0.0628 | 0.0642 |
| dense | text | 0.2944 | 0.2972 | 0.2997 |
| dense | messages | 0.3083 | 0.3130 | 0.3130 |
| hybrid | text | 0.2042 | 0.2097 | 0.2109 |
| hybrid | messages | 0.1986 | 0.2063 | 0.2092 |
