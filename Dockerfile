from tensorflow/tensorflow:2.10.0-gpu

RUN apt update && \
    apt install -y git && \
    pip install --no-cache-dir scikit-image==0.19.3 \
    ftfy==6.1.1 torch nest-asyncio==1.5.6 transformers diffusers accelerate pyngrok\
    fastapi "uvicorn[standard]" 

WORKDIR /app

COPY ./app.py /app/app.py
#uvicorn.
CMD uvicorn --host 0.0.0.0 app:app 