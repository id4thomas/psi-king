# pgvector ko-hybrid-v2
pgvector + pg_textsearch + textsearch_ko
- [pg_textsearch](https://github.com/timescale/pg_textsearch): provides bm25
- [textsearch_ko](https://github.com/i0seph/textsearch_ko): Korean full text search extension (uses mecab for tokenization)


Test Environment:
- DGX Spark (Asus GX10 - aarch64)

## Usage
### 

```
(base) yrlab@gx10-48c3:~/git/psi-king/experiments/korean_db/pgvector/ko-hybrid-v2$ docker exec -it hardcore_euclid psql -U id4thomas -d psi_king -c "SELECT cfgname FROM pg_ts_config WHERE cfgname = 'korean';"
 cfgname 
---------
 korean
(1 row)

```