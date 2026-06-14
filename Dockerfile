FROM python:3.11-slim

WORKDIR /app

# Copy your streamlit app files
COPY . /app

# If project_folder is outside your streamlit directory locally,
# you must explicitly copy it into the container image:
COPY ./project_folder /app/project_folder

# Install standard Linux build tools required by Pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file from the root folder
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# DO NOT hardcode ENV PORT. Cloud Run will inject this automatically.
EXPOSE 8080

# Use a shell execution command to dynamically read the Cloud Run $PORT variable
CMD ["sh", "-c", "streamlit run scripts/src/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]