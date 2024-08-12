from datetime import datetime, UTC
import json
from fastapi import FastAPI, Response, status as StatusCode
from fastapi.routing import APIRouter
import uvicorn
from routers.message import router as MESSAGE_ROUTER


app = FastAPI()
startup_time = datetime.now(tz=UTC)


@app.get('/')
def root():
    request_time = datetime.now(tz=UTC)
    return Response(
        content=json.dumps({
            'startup_time': startup_time.isoformat(),
            'uptime_seconds': (request_time - startup_time).total_seconds() // 1,
        }),
        status_code = StatusCode.HTTP_200_OK,
    )


core_router = APIRouter(prefix='/v1')
core_router.include_router(MESSAGE_ROUTER)
app.include_router(core_router)


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8080)
