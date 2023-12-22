


"""
파이썬으로 백엔드 서버 프로그램을 만드는 중.

각각 uri 별로 request 값과 result 값이 아래와 같은 서버 프로그램 코드를 작성하고  스웨거를 적용시켜줘.
flask-restx 를 사용 할 것

/new_token

request : {
  db : integer
}
result : {
  token: string
}


/prompt

request : {
  token: string
  prompt: string
}

result : {
  result: string
}

"""
import threading
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import uuid
import asyncio

import os
import sys
import logging
import openai
from langchain.chat_models import ChatOpenAI
from langchain_E_retrieval_tool import get_tools
from langchain.agents.agent_toolkits import (
  create_retriever_tool,
  create_conversational_retrieval_agent
)
from langchain.agents.agent import AgentExecutor
from langchain.tools import Tool
from langchain.schema.vectorstore import (
  VectorStore,
  VectorStoreRetriever
)
from langchain.docstore.document import Document
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import utils

load_dotenv()

logging.basicConfig(level='INFO')

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION")
sys.path.append(os.getenv("PYTHONPATH"))
llm_model = "gpt-3.5-turbo-1106"

llm = ChatOpenAI(model_name=llm_model, temperature=0)


is_debug = True
app = FastAPI(debug=is_debug, docs_url="/api-docs")

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인을 허용하는 예시 설정
    allow_credentials=True,
    allow_methods=["*"],  # 모든 메소드를 허용하는 예시 설정
    allow_headers=["*"],  # 모든 헤더를 허용하는 예시 설정
)

class ListOutput(BaseModel):
  list: list[dict]

class SummaryOutput(BaseModel):
  summary: str

class TokenOutput(BaseModel):
  token: str


class PromptRequest(BaseModel):
  token: str
  prompt: str


class PromptResult(BaseModel):
  result: str




class DBInfo:
  idx: int
  name: str
  title: str
  summary: str
  tool: Tool
  executor: AgentExecutor
  description: str
  file: str

  def __init__(self, idx: int, name:str, title:str, description: str, file: str):
    self.idx = idx
    self.name = name
    self.title = title
    self.description = description
    self.file = file
    # self.executor = None
    self.tool = None
    self.summary = None

  def get_summary(self) -> str:
    if self.summary is not None:
      return self.summary

    self.summary = utils.load_summary_from_file(llm, self.file)

    return self.summary

  def get_tool(self) -> Tool:
    if self.tool is not None:
      return self.tool

    retriever = utils.load_vectordb_from_file(self.file).as_retriever()
    if not isinstance(retriever, VectorStoreRetriever):
      raise ValueError("it's not VectorStoreRetriever")

    self.tool = create_retriever_tool(
      retriever,
      self.name,
      self.description
    )

    self.tools = [self.tool]

    return self.tool

  def new_executor(self) -> AgentExecutor:
    # if self.executor is not None:
    #   return self.executor

    executor = create_conversational_retrieval_agent(llm, [self.get_tool()], verbose=True)
    return executor


db_infos: list[DBInfo] = [
  DBInfo(
    1,
    "freelancer_guidelines",
    "프리랜서 가이드라인",
    "Good for answering questions about the different things you need to know about being a freelancer",
    "./data/프리랜서 가이드라인 (출판본).pdf"
  ),
  DBInfo(
    2,
    "colour_tarot",
    "마음의 비밀코드, 색채타로",
    "good for answering questions about colour tarot.",
    "./data/마음의비밀코드색채타로.pdf"
  ),
  DBInfo(
    3,
    "saju",
    "사주명리학 운세변화와 대운활용",
    "사주명리학에 기초해서 운세변화와 대운활용에 대한 질문에대해 답하기 좋다.",
    "./data/사주명리학 운세변화,대운활용(최종본)-09.2.17.pdf"
  ),
  DBInfo(
    4,
    "good_writer_reader_child",
    "말 잘 하는 아이와 글 잘 쓰는 아이",
    "말 잘 하는 아이와 글 잘 쓰는 아이에 대한 질문에 답하기 좋다.",
    "./data/말잘하는아이글 잘 쓰는 아이_본문.pdf"
  )
]


def find_db_info(db:int)->DBInfo:
  for db_info in db_infos:
    if db_info.idx == db:
      return db_info

  return None




# @app.get("/")
# async def serve_html():
#   return FileResponse('./html-docs/index.html')
class ExecutorInfo:
  db_info: DBInfo
  executor: AgentExecutor

  def __init__(self, db_info: DBInfo):
    self.db_info = db_info
    self.executor = None

  def get_executor(self) -> AgentExecutor:
    if self.executor is None:
      self.executor = self.db_info.new_executor()

    return self.executor



tokens = {

}

@app.get("/api/get_list")
async def get_list():
  logging.info("called - /api/get_list")
  l: list[dict] = [ {"idx": info.idx, "title": info.title} for info in db_infos]
  return jsonable_encoder(ListOutput(list=l))

@app.get("/api/new_token")
async def new_token(db: int):
  logging.info(f"called - /api/new_token - db={db}")
  # 원하는 db 처리 로직을 여기에 추가하실 수 있습니다.
  token=str(uuid.uuid4())

  db_info = find_db_info(db)

  tokens[token] = ExecutorInfo(db_info)
  return jsonable_encoder(TokenOutput(token=token))

request_idx = 0

# tools = get_tools()
# agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)

def get_summary_prompt(summary: str, db_info: DBInfo) -> str:
#   prompt = f"""
# The following is a [summary] of the '{db_info.title}'.
# Based on this summary, print out the following.

# print out a Korean translation of the summary.

# recommend 5 questions that people might ask.

# format it as follows
# <Korean translation>.
# -----
# 다음과 같은 질문이 가능합니다.
# <Question 1>.
# <Question 2>.
# <... List questions up to 5>


# [summary]
# {summary}
# """
  prompt = f"""
The following is a [summary] of the '{db_info.title}'.
print out a Korean translation of the summary.

[summary]
{summary}
"""
  return prompt

@app.get("/api/summary")
async def process_summary(token: str):
  info: ExecutorInfo = tokens[token]
  logging.info(f"called - /api/summary - token={token}")
  if not info:
    raise ValueError("token이 없습니다.")

  summary= info.db_info.get_summary()
  executor = info.get_executor()
  result = executor({"input": get_summary_prompt(summary, info.db_info)})

  return jsonable_encoder( SummaryOutput(summary=result["output"]) )

  # return jsonable_encoder( SummaryOutput(summary=summary) )


@app.post("/api/prompt")
async def process_prompt(request: PromptRequest):
  # 비동기적으로 처리할 내용을 여기에 구현합니다.
  # 예를 들어, 외부 API 호출이나 무거운 계산 작업 등을 비동기로 수행할 수 있습니다.
  global request_idx
  idx = request_idx
  request_idx = request_idx + 1

  logging.info(f"called - /api/prompt - token={request.token}, prompt='{request.prompt}'")
  info: ExecutorInfo = tokens[request.token]
  if not info:
    raise ValueError("token이 없습니다.")

  executor = info.get_executor()
  result = executor({"input": request.prompt})

  return jsonable_encoder(PromptResult(result=result["output"]))


app.mount("/", StaticFiles(directory="./html-docs", html=True), name="static")


if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=5000)
