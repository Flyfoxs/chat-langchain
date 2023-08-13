"""Main entrypoint for the app."""

import pickle

from langchain import OpenAI
from langchain.agents import initialize_agent

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from langchain.agents import Tool

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
    import langchain as lc
    from langchain import OpenAI, SQLDatabaseChain
    import os
    # from langchain_experimental.sql import SQLDatabaseChain
    # site-packages\langchain_experimental\sql\base.py
    # /opt/conda/lib/python3.9/site-packages/langchain_experimental/sql/base.py
    from sqlalchemy import processors

    port = 10000
    #os.environ["OPENAI_API_KEY"] = "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"
    include_tables = ["test_del"]
    con = lc.SQLDatabase.from_uri("hive://localhost:10000/abc?auth=NOSASL", include_tables = include_tables, schema="abc")
    #con = lc.SQLDatabase(engine)
    llm = OpenAI(temperature=0)
    db_chain = SQLDatabaseChain.from_llm(llm=llm, db=con, verbose=True,
                                         #prompt=PromptTemplate.from_template(prompt)
                                         )
    return db_chain
    # query = "how many records are there in this table?"
    # #query = "describe the test_del table"
    # db_chain.run(query)

local_tool = Tool(
    name='Query information about Ubix Company',
    func=qa_chain.run,
    description='Query information about Ubix Company'
)

db_tool = Tool(
    name='Query Info about Data',
    func=get_db_chain().run,
    description='Query Info about Data in table, Hive or Spark'
)

tools = [local_tool, db_tool]

llm = OpenAI(temperature=0)
zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
)

result = zero_shot_agent.acall(
    {"question": "who am I", "chat_history": chat_history}
)

print(f'result:{result}')