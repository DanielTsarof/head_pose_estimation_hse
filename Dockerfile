FROM python:3.10
LABEL authors="head-pose-app"

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8501

WORKDIR /app
COPY . .

#CMD ["streamlit", "run", "run_demo.py"]
ENTRYPOINT ["streamlit", "run", "run_demo.py", "--server.port=8501", "--server.address=0.0.0.0"]