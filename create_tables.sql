CREATE EXTENSION IF NOT EXISTS vector;

CREATE TYPE genre_type AS ENUM (
    'action',
    'adventure',
    'animation',
    'anime',
    'british_shows',
    'children',
    'classic',
    'comedies',
    'comedy',
    'crime',
    'crime_shows',
    'cult',
    'documentaries',
    'documentary',
    'docuseries',
    'drama',
    'dramas',
    'fantasy',
    'film_noir',
    'horror',
    'imax',
    'independent',
    'international',
    'international_shows',
    'kids',
    'korean_shows',
    'lgbtq',
    'musical',
    'mysteries',
    'mystery',
    'reality',
    'romance',
    'romantic',
    'romantic_shows',
    'sci_fi',
    'shows',
    'spanish_language_shows',
    'sports',
    'stand_up_comedy',
    'teen_shows',
    'thriller',
    'thrillers',
    'war',
    'western'
);

CREATE TABLE IF NOT EXISTS fixed_size_splitter_all_MiniLM_L6_v2 (
    id SERIAL PRIMARY KEY,
    title TEXT,
    year INTEGER,
    genres genre_type ARRAY,
    feature_vector VECTOR(384)
);

CREATE TABLE IF NOT EXISTS fixed_size_splitter_gtr_t5_base (
    id SERIAL PRIMARY KEY,
    title TEXT,
    year INTEGER,
    feature_vector VECTOR(768),
    genres genre_type ARRAY
);

CREATE TABLE IF NOT EXISTS fixed_size_splitter_bert_base_nli_mean_tokens (
    id SERIAL PRIMARY KEY,
    title TEXT,
    year INTEGER,
    genres genre_type ARRAY,
    feature_vector VECTOR(768)
);

CREATE TABLE IF NOT EXISTS recursive_splitter_all_MiniLM_L6_v2 (
    id SERIAL PRIMARY KEY,
    title TEXT,
    year INTEGER,
    genres genre_type ARRAY,
    feature_vector VECTOR(384)
);

CREATE TABLE IF NOT EXISTS recursive_splitter_gtr_t5_base (
    id SERIAL PRIMARY KEY,
    title TEXT,
    year INTEGER,
    feature_vector VECTOR(768),
    genres genre_type ARRAY
);

CREATE TABLE IF NOT EXISTS recursive_splitter_bert_base_nli_mean_tokens (
    id SERIAL PRIMARY KEY,
    title TEXT,
    year INTEGER,
    genres genre_type ARRAY,
    feature_vector VECTOR(768)
);

CREATE TABLE IF NOT EXISTS semantic_splitter_all_MiniLM_L6_v2 (
    id SERIAL PRIMARY KEY,
    title TEXT,
    year INTEGER,
    genres genre_type ARRAY,
    feature_vector VECTOR(384)
);

CREATE TABLE IF NOT EXISTS semantic_splitter_gtr_t5_base (
    id SERIAL PRIMARY KEY,
    title TEXT,
    year INTEGER,
    feature_vector VECTOR(768),
    genres genre_type ARRAY
);

CREATE TABLE IF NOT EXISTS semantic_splitter_bert_base_nli_mean_tokens (
    id SERIAL PRIMARY KEY,
    title TEXT,
    year INTEGER,
    genres genre_type ARRAY,
    feature_vector VECTOR(768)
);
