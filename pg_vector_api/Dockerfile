FROM python:3.11-slim

COPY ./*.py ./genres.txt ./requirements.txt /opt/app/
COPY embeddings /opt/app/embeddings

RUN pip install -r /opt/app/requirements.txt

EXPOSE 8080
WORKDIR /opt/app/

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
