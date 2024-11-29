FROM python:3.11.9

RUN pip install --upgrade pip

WORKDIR /usr/src/app

RUN pip install Flask
RUN pip install tensorflow
RUN pip install flask_cors
RUN pip install flwr
RUN pip install numpy
RUN pip install Pillow
RUN pip install typer
RUN pip install protobuf
RUN pip install rembg
RUN pip install Requests
RUN pip install onnxruntime

COPY ./backend ./backend

EXPOSE 5000
CMD ["python", "-m", "backend.main", "--mode", "api"]