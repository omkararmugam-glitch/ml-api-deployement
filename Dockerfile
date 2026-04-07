FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
