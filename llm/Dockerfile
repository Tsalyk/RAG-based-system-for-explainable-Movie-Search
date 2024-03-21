FROM python:3.11-slim

COPY ./*.py ./.env ./requirements.txt /opt/app/

RUN pip install -r /opt/app/requirements.txt

EXPOSE 8085

WORKDIR /opt/app/

CMD ["uvicorn", "llm_api:app", "--host", "0.0.0.0", "--port", "8085"]
