from tensorflow/tensorflow:2.10.0-gpu

RUN apt update && \
    apt install -y git && \
    pip install --no-cache-dir scikit-image==0.19.3 Pillow==9.2.0 tqdm==4.64.1\
    ftfy==6.1.1 regex==2022.9.13 torch transformers diffusers\
    fastapi "uvicorn[standard]" git+https://github.com/divamgupta/stable-diffusion-tensorflow.git

WORKDIR /app

COPY ./app.py /app/app.py

CMD uvicorn --host 0.0.0.0 app:app