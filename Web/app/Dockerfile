FROM python:3.11.10

RUN adduser nonroot
RUN apt install curl # for healthcheck

RUN mkdir /home/app/ && chown -R nonroot:nonroot /home/app
RUN mkdir -p /var/log/app && chown -R nonroot:nonroot /var/log/app

WORKDIR /home/app
USER nonroot
COPY --chown=nonroot:nonroot . .

# python setup
ENV VIRTUAL_ENV=/home/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m venv $VIRTUAL_ENV
# RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 6969
CMD ["python3", "main.py"]
