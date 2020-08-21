#! /bin/sh

if [ -z "$1" ]
then

echo "Starting dev server"
python app.py dev
else

echo "Starting dev docker container"
docker run --detach \
--publish 8000:8000 \
--workdir /app \
--volume "$(pwd):/app" \
--memory 1GB \
--name fusion-vision-dev \
python:3.6-slim-buster \
sh -c "python -m pip install -r requirements.txt && python app.py dev"

fi