from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from chunks import Chunk

app = FastAPI()
chunk = Chunk()

request_count = 0

# Класс с типами данных для метода get_answer
class ModelAnswer(BaseModel):
    text: str
    
@app.get('/')
def get_root():
    return {'message': 'Привет, я консультант по страхованию ответственности аэропортов и авиапроизводителей'}

# Маршрут для favicon.ico
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse('static/favicon.ico')

# Функция обработки запроса к нейро-консультанту
@app.post('/get_answer')
def get_answer(question: ModelAnswer):
    global request_count
    request_count += 1 
    answer = chunk.get_answer(query=question.text)
    return {'message': answer}

# Метод для получения количества обращений
@app.get("/get_count")
def get_count():
    global request_count
    return {"total_requests": request_count}