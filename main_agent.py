"""Main entrypoint for the app."""

import pickle
import os
from langchain import OpenAI
from langchain.agents import initialize_agent

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from langchain.agents import Tool
from sqlalchemy import create_engine

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse
with open("vectorstore.pkl", "rb") as f:
    global vectorstore
    vectorstore = pickle.load(f)

import asyncio
import websockets

async def echo(websocket, path):
    async for message in websocket:
        await websocket.send(message)

websocket: WebSocket = websockets.serve(echo, "localhost", 8765)

question_handler = QuestionGenCallbackHandler(websocket)
stream_handler = StreamingLLMCallbackHandler(websocket)

chat_history = []
qa_chain = get_chain(vectorstore, question_handler, stream_handler)




def get_db_chain():
    port = 10000
    include_tables = ["test_del"]
    from langchain import SQLDatabaseChain, OpenAI, SQLDatabase
    db = SQLDatabase(create_engine("hive://localhost:10000/abc?auth=NOSASL"), include_tables=include_tables)
    db_chain = SQLDatabaseChain(llm=OpenAI(), database=db, top_k=3, return_direct=True)
    return db_chain


local_tool = Tool(
    name='Query information about Ubix Company',
    func=qa_chain.run,
    description='Query information about me and Ubix Company',
)
local_tool.return_direct = True

db_tool = Tool(
    name='Query Info about Data',
    func=get_db_chain().run,
    description='Query Info about Data in table, Hive or Spark'
)
db_tool.return_direct = True

tools = [local_tool, db_tool]
llm = OpenAI(temperature=0)
zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
)

result = zero_shot_agent("Who am I?")
result = zero_shot_agent("Could you help to count how many rows are there in the table test_del?")

# print(f'result:{result}')
