FROM mybase:latest

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir imageio

CMD ["tail", "-f", "/dev/null"]