# 🛠 MLOps Set Up Guide
## PostgreSQL service
This database service is used to stored data for mlflow's backend, auth and lakeFS metadata, thus three additional databases are going to created at database initiating period ([check this](./postgres/init-user-db.sh)).

## LakeFS service
This service use two different types of database for operating
* PL/pgSQL: stores metadata.
* Minio: store object like S3 (AWS).  

Instead of using isolated .lakefs.yaml configuration file, we embed lakeFS's configurations into [this](./lakeFS/generate_config.sh), which is executed when initiating docker compose up.  

## Mlflow service
In order to restrain the access from unknown inbound request, we enabled experimental HTTP authentication feature of mlflow ([check this](./MLflow/basic_auth.ini)). In addition, we also provide a short sql script for removing experiment in deleted status.


## How to run
```bash
docker compose up --build
```
