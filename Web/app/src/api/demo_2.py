from fastapi import Response
from fastapi.routing import APIRouter


router = APIRouter()


@router.get("/health-check")
def check_health():
    return {
        "exit code": Response(status_code=200).status_code,
        "message": "Demo api 2. Check health"
    }
