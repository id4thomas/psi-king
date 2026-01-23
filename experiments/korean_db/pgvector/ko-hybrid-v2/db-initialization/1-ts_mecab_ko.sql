-- 1. 스키마 경로 설정
SET search_path = public;

-- 2. Mecab Parser 관련 함수 등록
-- OR REPLACE를 사용하여 기존 함수가 있어도 덮어씁니다.
CREATE OR REPLACE FUNCTION ts_mecabko_start(internal, int4)
    RETURNS internal
    AS '$libdir/ts_mecab_ko'
    LANGUAGE 'c' STRICT;

CREATE OR REPLACE FUNCTION ts_mecabko_gettoken(internal, internal, internal)
    RETURNS internal
    AS '$libdir/ts_mecab_ko'
    LANGUAGE 'c' STRICT;

CREATE OR REPLACE FUNCTION ts_mecabko_end(internal)
    RETURNS void
    AS '$libdir/ts_mecab_ko'
    LANGUAGE 'c' STRICT;

CREATE OR REPLACE FUNCTION ts_mecabko_lexize(internal, internal, internal, internal)
    RETURNS internal
    AS '$libdir/ts_mecab_ko'
    LANGUAGE 'c' STRICT;

-- 3. Parser, Template, Dictionary, Configuration 등록
-- 이미 존재하는 경우 에러가 나지 않도록 DO 블록 내부에서 조건부로 생성합니다.
DO $$
BEGIN
    -- (1) Korean Text Parser 등록
    IF NOT EXISTS (SELECT 1 FROM pg_ts_parser WHERE prsname = 'korean') THEN
        CREATE TEXT SEARCH PARSER korean (
            START    = ts_mecabko_start,
            GETTOKEN = ts_mecabko_gettoken,
            END      = ts_mecabko_end,
            HEADLINE = pg_catalog.prsd_headline,
            LEXTYPES = pg_catalog.prsd_lextype
        );
        COMMENT ON TEXT SEARCH PARSER korean IS 'korean word parser';
    END IF;

    -- (2) Text Search Template 등록
    IF NOT EXISTS (SELECT 1 FROM pg_ts_template WHERE tmplname = 'mecabko') THEN
        CREATE TEXT SEARCH TEMPLATE mecabko (
            LEXIZE = ts_mecabko_lexize
        );
    END IF;

    -- (3) Text Search Dictionary 등록
    IF NOT EXISTS (SELECT 1 FROM pg_ts_dict WHERE dictname = 'korean_stem') THEN
        CREATE TEXT SEARCH DICTIONARY korean_stem (
            TEMPLATE = mecabko
        );
    END IF;

    -- (4) 가장 중요한 'korean' Text Search Configuration 등록
    IF NOT EXISTS (SELECT 1 FROM pg_ts_config WHERE cfgname = 'korean') THEN
        CREATE TEXT SEARCH CONFIGURATION korean (PARSER = korean);
        COMMENT ON TEXT SEARCH CONFIGURATION korean IS 'configuration for korean language';

        -- 기본 매핑 설정 (숫자, URL, 이메일 등)
        ALTER TEXT SEARCH CONFIGURATION korean ADD MAPPING
            FOR email, url, url_path, host, file, version,
                sfloat, float, int, uint,
                numword, hword_numpart, numhword
            WITH simple;

        -- 영문 매핑 (english_stem 사용)
        ALTER TEXT SEARCH CONFIGURATION korean ADD MAPPING
            FOR asciiword, hword_asciipart, asciihword
            WITH english_stem;

        -- 한글 매핑 (korean_stem 사용)
        ALTER TEXT SEARCH CONFIGURATION korean ADD MAPPING
            FOR word, hword_part, hword
            WITH korean_stem;
    END IF;
END $$;

-- 4. 유틸리티 함수 등록
CREATE OR REPLACE FUNCTION mecabko_analyze(
        text,
        OUT word text,
        OUT type text,
        OUT part1st text,
        OUT partlast text,
        OUT pronounce text,
        OUT conjtype text,
        OUT conjugation text,
        OUT basic text,
        OUT detail text,
        OUT lucene text)
    RETURNS SETOF record
    AS '$libdir/ts_mecab_ko'
    LANGUAGE 'c' IMMUTABLE STRICT;

CREATE OR REPLACE FUNCTION korean_normalize(text)
    RETURNS text
    AS '$libdir/ts_mecab_ko'
    LANGUAGE 'c' IMMUTABLE STRICT;

CREATE OR REPLACE FUNCTION hanja2hangul(text)
    RETURNS text
    AS '$libdir/ts_mecab_ko'
    LANGUAGE 'c' IMMUTABLE STRICT;