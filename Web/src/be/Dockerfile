FROM python:3.11.10

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y curl && apt clean

RUN adduser nonroot
RUN mkdir -p /Web/src/be && chown -R nonroot:nonroot /Web/src/be

COPY --chown=nonroot:nonroot . /Web/src/be

# Directly install requirements without venv
ENV VIRTUAL_ENV=/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3.11 -m venv $VIRTUAL_ENV
RUN pip install --no-cache-dir -r /Web/src/be/requirements.txt

EXPOSE 6967
CMD ["python3", "/Web/src/be/main.py"]