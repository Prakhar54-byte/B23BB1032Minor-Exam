FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir datasets

COPY load_data.py .

CMD ["python", "b23bb1032_data_loader.py"]