# Text Deduplication
* retrieval 결과에 대해 중복 제거를 위해 deduplication 과정을 거침


## Resources
### 설명
NearDedup
- 스캐터랩 자료
    - https://tech.scatterlab.co.kr/deduplication/
    - Google NearDedup 방식 사용
- (Google ACL 2022) Deduplicating Training Data Makes Language Models Better
    - https://github.com/google-research/deduplicate-text-datasets
    - NearDedup 알고리즘: 중복일 가능성이 매우 높은 쌍들에 대해서만 편집 거리를 계산
    - LSH 알고리즘 → 단어 집합 Jjacard Similarity 기반 필터링 → 편집거리 계산

Editdistance
- [https://velog.io/@49crehbgr/알고리즘Algorithm-편집거리-알고리즘](https://velog.io/@49crehbgr/%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98Algorithm-%ED%8E%B8%EC%A7%91%EA%B1%B0%EB%A6%AC-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98)

LSH (Locality-Sensitive Hashing) 알고리즘
- https://lifemath.tistory.com/8

### Packages
- PolyDeDupe: multilingual dedup
    - https://github.com/gagan3012/PolyDeDupe
- https://github.com/ChenghaoMou/text-dedup

