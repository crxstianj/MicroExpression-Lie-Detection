
FROM python:3.10

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
