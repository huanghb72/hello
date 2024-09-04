# hello

## Setup

Python version: 3.9

Install virtualenv, create a new venv and run:

pip install -r requirements.txt

## Testing API

In the /asr folder, run:

uvicorn --port 8001 asr_api:app

## Docker

In the /asr folder, run:

docker build . -t asr-api

docker compose up
