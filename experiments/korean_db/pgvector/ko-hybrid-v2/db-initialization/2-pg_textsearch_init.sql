SET search_path = public;

BEGIN;

-- 1. pg_textsearch 확장 프로그램 활성화 (BM25 기능)
-- CREATE EXTENSION IF NOT EXISTS pg_textsearch;
CREATE EXTENSION IF NOT EXISTS pg_textsearch SCHEMA public;

-- 2. (선택) RRF(Reciprocal Rank Fusion) 점수 계산 함수 정의
-- 앱 코드에서 매번 복잡한 산식을 쓰지 않고 함수로 호출할 수 있습니다.
CREATE OR REPLACE FUNCTION calculate_rrf(
    semantic_rank integer, 
    keyword_rank integer, 
    k integer DEFAULT 60
)
RETURNS float8 AS $$
BEGIN
    RETURN (COALESCE(1.0 / (k + semantic_rank), 0.0) + 
            COALESCE(1.0 / (k + keyword_rank), 0.0));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON EXTENSION pg_textsearch IS 'BM25 scoring for full-text search';
COMMENT ON FUNCTION calculate_rrf IS 'Calculates Reciprocal Rank Fusion score from two ranks';

COMMIT;