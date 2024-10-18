FROM python:3.12.7-slim-bookworm
WORKDIR /app
COPY . /app
RUN apt update -y && pip install -r requirements.txt
CMD streamlit run app.py --server.port 8500