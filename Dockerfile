# syntax=docker/dockerfile:1

FROM python:3.8
COPY requirements.txt /requirements.txt
WORKDIR /bin/src

# install app dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install -r requirements.txt

# install app
COPY . .
EXPOSE 8000
CMD flask run --host 0.0.0.0 --port 8000