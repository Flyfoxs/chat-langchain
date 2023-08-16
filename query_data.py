"""Create a ChatVectorDBChain for question/answering."""
import pickle

import prompt_toolkit
from langchain import PromptTemplate
from langchain.agents import Tool, initialize_agent
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain, ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores.base import VectorStore
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain import SQLDatabaseChain, OpenAI, SQLDatabase

from typing import Any, Callable, Dict, List, Optional, Tuple, Union


from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores.base import VectorStore
from sqlalchemy import create_engine

class ConversationalRetrievalChainExt(ConversationalRetrievalChain):

    def prep_inputs(self, inputs: str):
        return super().prep_inputs({"question": inputs, "chat_history": []})

def get_kb_chain(
    vectorstore: VectorStore, tracing: bool = False
) -> ConversationalRetrievalChain:
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
    #
    # question_gen_llm = OpenAI(
    #     temperature=0,
    #     verbose=True,
    #     # callback_manager=question_manager,
    # )
    # streaming_llm = OpenAI(
    #     streaming=True,
    #     # callback_manager=stream_manager,
    #     verbose=True,
    #     temperature=0,
    # )
    #
    # question_generator = LLMChain(
    #     llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    # )
    # doc_chain = load_qa_chain(
    #     streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
    # )



    from langchain.chains import (
        StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
    )
    from langchain.prompts import PromptTemplate
    from langchain.llms import OpenAI

    combine_docs_chain = get_stuff_chain()
    # vectorstore = ...
    retriever = vectorstore.as_retriever()

    # This controls how the standalone question is generated.
    # Should take `chat_history` and `question` as input variables.
    template = (
        "Combine the chat history and follow up question into "
        "a standalone question. Chat History: {chat_history}"
        "Follow up question: {question}"
    )
    prompt = PromptTemplate.from_template(template)
    llm = OpenAI()
    question_generator_chain = LLMChain(llm=llm, prompt=prompt)
    qa = ConversationalRetrievalChainExt(
        combine_docs_chain=combine_docs_chain,
        retriever=retriever,
        question_generator=question_generator_chain,
        callback_manager=manager,
    )

    # qa = ChatVectorDBChainExt(
    #     vectorstore=vectorstore,
    #     combine_docs_chain=doc_chain,
    #     question_generator=question_generator,
    #     callback_manager=manager,
    # )
    return qa


def get_stuff_chain():
    from langchain.chains import StuffDocumentsChain, LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.llms import OpenAI

    # This controls how each document will be formatted. Specifically,
    # it will be passed to `format_document` - see that function for more
    # details.
    document_prompt = PromptTemplate(
        input_variables=["page_content"],
        template="{page_content}"
    )
    document_variable_name = "context"
    llm = OpenAI()
    # The prompt here should take as an input variable the
    # `document_variable_name`
    prompt = PromptTemplate.from_template(
        "Summarize this content: {context}"
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name
    )
    return chain

def get_db_chain():
    import langchain as lc
    include_tables = ["test_del"]
    con = lc.SQLDatabase.from_uri("hive://localhost:10000/abc?auth=NOSASL", include_tables = include_tables, schema="abc")
    llm = OpenAI(temperature=0)
    db_chain = SQLDatabaseChain.from_llm(llm=llm, db=con, verbose=True,
                                         #prompt=PromptTemplate.from_template(prompt)
                                         )
    return db_chain


def get_general_chain():
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""{question}"""
            )
    llm = OpenAI(temperature=0)
    search_chain = LLMChain(llm=llm, prompt=prompt)
    search_tool = Tool(
        name='General Question',
        func=search_chain.run,
        description='General Question'
    )
    search_tool.return_direct = True
    return search_tool


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

    tools = [local_tool
        , db_tool
        # , get_general_chain()
             ]
    return tools

def get_agent(websocket=None, question_handler=None, stream_handler=None):
    tools = get_tools()
    llm = OpenAI(temperature=0)
    manager = AsyncCallbackManager([])
    if True:
        tracer = LangChainTracer()
        # tracer.load_default_session()
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
    import os

    zero_shot_agent = get_agent()
    result = zero_shot_agent("Who is Ubix?")
    result = zero_shot_agent("Could you help to count how many rows are there in the table test_del?")

    #
    # from langchain.memory import ConversationBufferMemory
    # from langchain.chat_models import ChatOpenAI
    #
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # llm = ChatOpenAI(temperature=0)
    # agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
