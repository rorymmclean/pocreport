### Imports
import streamlit as st
import os
import io
from datetime import datetime
import chromadb
import langchain
# from langchain.cache import InMemoryCache
# langchain.llm_cache = InMemoryCache()
# from langchain.cache import SQLiteCache
# langchain.llm_cache = SQLiteCache(database_path="langchain.db")
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, Tool, tool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.llms import OpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain import LLMMathChain
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import create_python_agent
from contextlib import redirect_stdout
from typing import Optional, Type
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

### Read runbook and create vector index
raw_documents = TextLoader('content/runbook.txt').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
runbook_vectors = Chroma.from_documents(documents, OpenAIEmbeddings())


### CSS
st.set_page_config(
    page_title='Report Chain', 
    layout="wide",
    initial_sidebar_state='collapsed',
)
padding_top = 1
st.markdown(f"""
    <style>
        .block-container, .main {{
            padding-top: {padding_top}rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# OpenAI Credentials
if not os.environ["OPENAI_API_KEY"]:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai_api_key = os.environ["OPENAI_API_KEY"]

### UI
col1, col2 = st.columns( [1,5] )
col1.image('AderasBlue2.png', width=70)
col1.image('AderasText.png', width=70)
col2.title('LLM Incident Report Chain')

def get_input() -> str:
    st.write("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
    contents = []
    # contents = st.text_input('here')
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        contents.append(line)
        st.write(contents)
    return "\n".join(contents)

def run_inquiry(myquestion):
    st.session_state.messages.append({"role": "user", "content": myquestion})
    st.chat_message("user").write(myquestion)

    ### Bring in my controlling documents and the additonal template
    relevancy_cutoff = .8
    
    docs = runbook_vectors.similarity_search_with_relevance_scores(myquestion, k=3)

    mytasks = ""
    for x,v in docs:
        if mytasks == "":
            mytasks = str(x) + "\n\n"
            continue
        if v > relevancy_cutoff:
            mytasks = mytasks + str(x) + "\n\n"

    template=f"""
    You are a helpful chatbot that collects information from the user based upon the type of report they need to comnplete.
    Collect information from the user based upon the report type referenced in the  "PROMPT" below. 
    Use the "TEXT" below to help develop to steps necessary to collect the correct information.
    If the users does not reference a report you understand, then return the answer: "Sorry, but I can't help you with that task."

    TEXT:
    {mytasks}

    PROMPT:
    {myquestion}
    """
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.0, verbose=False)
    tools = load_tools(
        ["human", "llm-math"],
        llm=llm,
        input_func=get_input,
    )

    executor = load_agent_executor(
        llm, 
        tools, 
        verbose=True,
    )

    planner = load_chat_planner(llm=llm)

    pe_agent = PlanAndExecute(
        planner=planner, 
        executor=executor,  
        verbose=True, 
        max_iterations=2,
        max_execution_time=180,
        AgentType = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    )

    if show_detail:
        f = io.StringIO()
        with redirect_stdout(f):
            # with st.spinner("Processing..."):
            response = pe_agent(template)
    else:
        # with st.spinner("Processing..."):
        response = pe_agent(template)

    st.session_state.messages.append({"role": "assistant", "content": response['output']})    
    st.chat_message('assistant').write(response['output'])

    if show_detail:
        with st.expander('Details', expanded=False):
            s = f.getvalue()
            st.write(s)

with st.sidebar: 
    show_detail = st.checkbox('Show Details')
    llm_model = st.selectbox('Select Model', ['gpt-4', 'gpt-3.5-turbo'])
    st.markdown("---")
    st.markdown("### Standard Questions:")
    bt1 = st.button('Burglary Crime Report')
    bt2 = st.button('Crime Scene Report')
    st.markdown("---")
    tz = st.container()

with st.expander("**:blue[Incident Report Chain Overvew]**"):
    st.markdown("**:blue[This LLM chain uses a Planner/Executor chain and a \"Human-as-a-Tool\" feature that allows ChatGPT to prompt the user for inputs needed for an incident report of some kind.]**")
    st.markdown("**:blue[The user enters a request to complete a certain type of incident report. The workflow first enters the Planner phase where it uses vector semantic KNN search of our \"run book\" document to determine what information is needed for the various types of reports. This document resembles an FAQ document and provides steps for completing these tasks. These are tasks a human might take. The LLM will translate these steps into steps that LangChain can execute.]**")
    st.markdown("**:blue[In the Executor phase the LLM instructs LangChain which tools to select and what tasks to perform. The most frequently used tool with be the \"Human-as-a-Tool\" feature so ChatGPT can prompt the user to provide more information. Other possible tools could include dynamic Python using SciKit, querying the internet, running shell scripts, running REST queries, or any act that might be defined in runbook. Chains are toolsets that must be assembled to meet a general area of inquiry.]**")

    col1, col2, col3 = st.columns([15, 70, 15])
    col2.image('chain.png',caption='LangChain Structure')

# st.markdown('---')

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "What incident report would you like to complete?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Ask a question?"):
    start = datetime.now()
    tz.write("Start: "+str(start)[10:])
    run_inquiry(prompt)
    tz.write("End: "+str(datetime.now())[10:])
    tz.write("Duration: "+str(datetime.now() - start))
if bt1:
    start = datetime.now()
    tz.write("Start: "+str(start)[10:])
    run_inquiry("Help me complete a burlary report")
    tz.write("End: "+str(datetime.now())[10:])
    tz.write("Duration: "+str(datetime.now() - start))
if bt2:
    start = datetime.now()
    tz.write("Start: "+str(start)[10:])
    run_inquiry("Help me complete a crime sceen report") 
    tz.write("End: "+str(datetime.now())[10:])
    tz.write("Duration: "+str(datetime.now() - start))
