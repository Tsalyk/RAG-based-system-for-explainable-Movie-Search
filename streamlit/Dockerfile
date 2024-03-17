FROM python:latest

COPY ./*.py ./data.csv ./.env ./requirements.txt /opt/app/

RUN pip install -r /opt/app/requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "/opt/app/app.py"]