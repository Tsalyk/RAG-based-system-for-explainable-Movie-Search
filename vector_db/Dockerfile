FROM python:3.11-slim

COPY ./*.py ./pinecone_db.py ./genres.txt ./requirements.txt /opt/app/

RUN pip install -r /opt/app/requirements.txt

EXPOSE 8080
WORKDIR /opt/app/

# ENTRYPOINT ["python", "/opt/app/api.py"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
