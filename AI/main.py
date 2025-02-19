import mlflow
import lakefs


from mlflow.data.http_dataset_source import HTTPDatasetSource
from lakefs_spec import LakeFSFileSystem




def main() -> None:
    # mlflow.log_input()



    # dataset_source_url = f"s3://{repo.id}/{repo.ref('caad29bddf4ea94fb6ae701a5c93d779edafdd86f5881c9a7c4e59d5266062dd')}/"
    # ds = HTTPDatasetSource(dataset_source_url)
    # ds.load("/home/trong/Downloads")
    return None

if __name__ == '__main__':
    main()