import json
from typing import  List

## File-Level
def calculate_filelevel_ap(
    retrieved_ids: List[str],
    answer_id: str,
    at_k: int = 5
) -> float:
    precisions = []
    relevant_count = 0
    for i in range(at_k):
        if len(retrieved_ids)<=i:
            precisions.append(
                relevant_count/(i+1)
            )
            continue
            
        if retrieved_ids[i]==answer_id:
            relevant_count+=1
        precisions.append(
            relevant_count/(i+1)
        )
    return sum(precisions)/len(precisions)
        
def calculate_filelevel_rr(
    retrieved_ids: List[str],
    answer_id: str,
    at_k: int = 5
) -> float:
    rr = 0.0
    for i in range(at_k):
        if len(retrieved_ids)<=i:
            return 0.0
            
        if retrieved_ids[i]==answer_id:
            return 1/(i+1)
    return rr

## Page-Level
def determine_page_level_relevancy(metadata, answer):
    answer_fileid = answer[0]
    answer_pageno = answer[1]
    
    retrieved_fileid = metadata['source_id']
    if answer_fileid!=retrieved_fileid:
        return 0
    
    if metadata['reader']=='DoclingPDFReader':
        prov = json.loads(metadata['prov'])
        page_nos = [
            x['page_no'] for x in prov
        ]
    else:
        prov = metadata['prov']
        page_nos = [
            prov['page']
        ]
    
    if answer_pageno in page_nos:
        return 1
    return 0

def calculate_pagelevel_ap(
    retrieved_metadatas: List[dict],
    answer: tuple,
    at_k: int = 5
) -> float:
    precisions = []
    relevant_count = 0
    for i in range(at_k):
        if len(retrieved_metadatas)<=i:
            precisions.append(
                relevant_count/(i+1)
            )
            continue
            
        metadata = retrieved_metadatas[i]
        if determine_page_level_relevancy(metadata, answer):
            relevant_count+=1
        precisions.append(
            relevant_count/(i+1)
        )
    return sum(precisions)/len(precisions)


def calculate_pagelevel_rr(
    retrieved_metadatas: List[dict],
    answer: tuple,
    at_k: int = 5
) -> float:
    for i in range(at_k):
        if len(retrieved_metadatas)<=i:
            return 0.0
            
        metadata = retrieved_metadatas[i]
        if determine_page_level_relevancy(metadata, answer):
            return 1/(i+1)
    return 0.0