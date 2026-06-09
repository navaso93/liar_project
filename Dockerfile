FROM python:3.10.6-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY liar liar
COPY api api
COPY saved_models saved_models

RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

CMD uvicorn api.fast:app --host 0.0.0.0 --port ${PORT:-8000}
