import json
from fastapi import Response, status as StatusCode
from fastapi.routing import APIRouter
# from pydantic import BaseModel
from llm.openai import TranslationRequest, TranslationResponse, chat_with_translation


router = APIRouter(prefix="/message")


@router.post('', response_model=TranslationResponse)
def create_message(req: TranslationRequest):
    res = chat_with_translation(req)
    return Response(
        content=json.dumps(res.to_dict(), ensure_ascii=False),
        status_code=StatusCode.HTTP_200_OK,
        media_type="application/json; charset=utf-8",
    )
