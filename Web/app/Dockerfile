FROM python:3.11.10

RUN adduser nonroot
RUN apt-get update && apt-get install -y curl && apt-get clean

RUN mkdir /home/app/ && chown -R nonroot:nonroot /home/app
RUN mkdir -p /var/log/app && chown -R nonroot:nonroot /var/log/app

WORKDIR /home/app
USER nonroot
COPY --chown=nonroot:nonroot . .

# Directly install requirements without venv
RUN pip install --no-cache-dir -r requirements.txt
    
EXPOSE 8000
CMD ["python3", "main.py"]