FROM python:3.11.9-alpine
RUN pip install --upgrade pip
COPY ./backend /app
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8080
ENTRYPOINT ["python"]
CMD ["-m", "backend.main", "--mode", "api"]