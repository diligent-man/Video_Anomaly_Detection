from fastapi import APIRouter, Response


router = APIRouter()


@router.get("/")
def say_hello():
    return {
      "exit code": Response(status_code=200).status_code,
      "message": "Demo api 1. Hello World"
    }


