FROM python:3.11-slim

COPY ./*.py ./data.csv ./.env ./requirements.txt /opt/app/

RUN pip install -r /opt/app/requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "/opt/app/app.py"]
