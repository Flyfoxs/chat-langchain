"""Create a ChatVectorDBChain for question/answering."""
import pickle

from langchain import SQLDatabaseChain
from langchain.agents import Tool, initialize_agent
from langchain.callbacks.base import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores.base import VectorStore



from typing import Any, Callable, Dict, List, Optional, Tuple, Union


from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores.base import VectorStore
from sqlalchemy import create_engine

class ChatVectorDBChainExt(ChatVectorDBChain):

    def prep_inputs(self, inputs: str):
        return super().prep_inputs({"question": inputs, "chat_history": []})

def get_kb_chain(
    vectorstore: VectorStore, tracing: bool = False
) -> ChatVectorDBChain:
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    # question_manager = AsyncCallbackManager([question_handler])
    # stream_manager = AsyncCallbackManager([stream_handler])
    # if tracing:
    #     tracer = LangChainTracer()
    #     tracer.load_default_session()
    #     manager.add_handler(tracer)
        # question_manager.add_handler(tracer)
        # stream_manager.add_handler(tracer)

    question_gen_llm = OpenAI(
        temperature=0,
        verbose=True,
        # callback_manager=question_manager,
    )
    streaming_llm = OpenAI(
        streaming=True,
        # callback_manager=stream_manager,
        verbose=True,
        temperature=0,
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
    )

    qa = ChatVectorDBChainExt(
        vectorstore=vectorstore,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
    )
    return qa


def get_db_chain():
    include_tables = ["test_del"]
    from langchain import SQLDatabaseChain, OpenAI, SQLDatabase
    db = SQLDatabase(create_engine("hive://localhost:10000/abc?auth=NOSASL"), include_tables=include_tables)
    db_chain = SQLDatabaseChain(llm=OpenAI(), database=db, top_k=3, return_direct=True)
    return db_chain


def get_tools():
    global vectorstore
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    qa_chain = get_kb_chain(vectorstore)
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
    return tools

def get_agent(websocket=None, question_handler=None, stream_handler=None):
    tools = get_tools()
    llm = OpenAI(temperature=0)
    manager = AsyncCallbackManager([])
    if True:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
    zero_shot_agent = initialize_agent(
        agent="zero-shot-react-description",
        tools=tools,
        llm=llm,
        verbose=True,
        # callback_manager=manager,
    )
    return zero_shot_agent


if __name__ == '__main__':
    zero_shot_agent = get_agent()
    result = zero_shot_agent("Who am I?")
    result = zero_shot_agent("Could you help to count how many rows are there in the table test_del?")
