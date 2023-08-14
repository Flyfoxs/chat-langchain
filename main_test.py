"""Main entrypoint for the app."""
import logging
import pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse
with open("vectorstore.pkl", "rb") as f:
    global vectorstore
    vectorstore = pickle.load(f)

import asyncio
import websockets
import os
async def echo(websocket, path):
    async for message in websocket:
        await websocket.send(message)

websocket: WebSocket = websockets.serve(echo, "localhost", 8765)

question_handler = QuestionGenCallbackHandler(websocket)
stream_handler = StreamingLLMCallbackHandler(websocket)

chat_history = []
qa_chain = get_chain(vectorstore, question_handler, stream_handler)
result = qa_chain(
    {"question": "who am I", "chat_history": chat_history}
)

print(f'result:{result}')
