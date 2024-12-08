FROM postgres:17.2
ARG DEBIAN_FRONTEND=noninteractive
COPY init-user-db.sh /docker-entrypoint-initdb.d
EXPOSE 5432
