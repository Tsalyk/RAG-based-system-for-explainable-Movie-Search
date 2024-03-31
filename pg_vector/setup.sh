until docker exec postgres-pgvector pg_isready -U admin; do
    sleep 1
done

docker exec -it postgres-pgvector psql -U admin -d sentence_embeddings -c 'CREATE EXTENSION vector'

docker exec -i postgres-pgvector psql -U admin -d sentence_embeddings < create_tables.sql
