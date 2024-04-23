FROM python:3.11-slim-buster

COPY ./*.py ./requirements.txt /opt/app/

RUN pip install --upgrade pip
RUN pip install -r /opt/app/requirements.txt


EXPOSE 8090

WORKDIR /opt/app/

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8090"]
