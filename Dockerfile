# tracking_system/Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY service_client.py demo_loop.py requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

# Por defecto lanza la demo; luego podrás ejecutar otros scripts con `docker compose exec`
CMD ["python", "main.py"]
