FROM python:3.10

WORKDIR /code

# Install ffmpeg
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg

# Install requirements
COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Cache the model so it does not need to be downloaded within the container
# Put huggingface/hub in ./.cache beforehand
# COPY ./.cache /cache

# ENV HF_HOME=/cache

# Copy code and run
COPY ./asr_api.py /code/asr_api.py

CMD ["fastapi", "run", "asr_api.py", "--port", "8001"]
