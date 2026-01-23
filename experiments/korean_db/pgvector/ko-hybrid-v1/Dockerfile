ARG POSTGRES_VERSION="16"
FROM pgvector/pgvector:pg${POSTGRES_VERSION}
ARG POSTGRES_VERSION

# https://www.bearpooh.com/136
RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    sed -i 's/# ko_KR.UTF-8 UTF-8/ko_KR.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen

ENV LANG=ko_KR.utf8 \
    LC_COLLATE=ko_KR.utf8 \
    LD_LIBRARY_PATH=/usr/local/lib \
    POSTGRES_INITDB_ARGS=--data-checksums

# Install Mecab
RUN set -eux; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
        build-essential \
        apt-utils \
        autoconf \
        ca-certificates \
        curl \
        tzdata \
        git \
        gcc \
        g++ \
        libtool \
        make \
        cmake \
        automake \
        zlib1g-dev; \
    apt-get install -y postgresql-server-dev-${POSTGRES_VERSION}; \
    ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime; \
    dpkg-reconfigure -f noninteractive tzdata; \
    apt-get autoremove; \
    apt-get clean;


ENV TZ=Asia/Seoul \
    LC_ALL=C.UTF-8 \
    SRC_PATH=/usr/local/src

RUN echo "Installing mecab-ko..."; \
    MECAB_DIST="mecab-0.996-ko-0.9.2"; \
    cd "$SRC_PATH"; \
    curl -s -LO https://bitbucket.org/eunjeon/mecab-ko/downloads/"$MECAB_DIST".tar.gz; \
    tar -xvzf "$MECAB_DIST".tar.gz; rm -f "$MECAB_DIST".tar.gz; cd "$MECAB_DIST"; \
    ./configure; \
    make all; \
    make install; \
    echo "mecab-ko install done"

# https://velog.io/@csu5216/nlp-docker로-mecab-ko-dic-설치시-에러
RUN echo "/usr/local/lib" >> /etc/ld.so.conf \
    && ldconfig

RUN set -eux; \
    \
    echo "Installing mecab-ko-dic"; \
    MD_DIST="mecab-ko-dic-2.1.1-20180720"; \
    cd "$SRC_PATH"; \
    curl -s -LO https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/"$MD_DIST".tar.gz; \
    tar -xvzf "$MD_DIST".tar.gz; rm -f "$MD_DIST".tar.gz; cd "$MD_DIST"; \
    ./autogen.sh; \
    ./configure; \
    make; make install; \
    echo "include ld.so.conf.d/*.conf" >> /etc/ld.so.conf; \
    echo "$LD_LIBRARY_PATH" >> /etc/ld.so.conf; \
    /sbin/ldconfig; \
    echo "mecab-ko-dic install done"

RUN echo "Building textsearch_ko extension"; \
    cd  "$SRC_PATH"; \
    git clone https://github.com/i0seph/textsearch_ko.git; \
    cd textsearch_ko; \
    make USE_PGXS=1; \
    make USE_PGXS=1 install

# psql -f ts_mecab_ko.sql

# Cleaning
RUN apt-get remove -y build-essential git wget autoconf