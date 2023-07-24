from tensorflow/tensorflow:2.10.0-gpu

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

WORKDIR /app

COPY ./app.py /app/app.py
#uvicorn.
CMD uvicorn --host 0.0.0.0 app:app 