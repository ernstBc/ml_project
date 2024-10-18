FROM python:3.12.7-slim-bookworm

WORKDIR /app

COPY . /app/

RUN apt update -y
RUN pip install -r requirements.txt

EXPOSE 8500

CMD streamlit run app.py --server.port 8500