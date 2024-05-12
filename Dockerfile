FROM python:3.11-slim

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED = 1

WORKDIR /app

RUN python -m pip install --upgrade pip

# Copy the source code into the container.
COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=deploymentRequirement.txt,target=deploymentRequirement.txt \ 
    python -m pip install -r deploymentRequirement.txt 

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 curl -y

# Expose the port that the application listens on.
EXPOSE 5000

#Run the app
CMD python app.py