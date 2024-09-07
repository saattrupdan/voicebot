FROM python:3.11-slim-bookworm

# Install Poetry
RUN pip install "poetry==1.8.2"

# Move the files into the container
WORKDIR /project
COPY . /project

# Install dependencies
RUN poetry env use python 3.11
RUN poetry install --no-interaction --no-cache --without dev

# Run the script
CMD poetry run python src/scripts/run_bot.py
