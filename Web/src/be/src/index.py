from fastapi import APIRouter, Response


router = APIRouter()

@router.get("/")
def read_root():
    return {
      "exit code": Response(status_code=200).status_code,
      "message": "This is root"
    }

@router.get("/health")
def read_root():
    return {
      "exit code": Response(status_code=200).status_code,
      "message": "This is root"
    }
