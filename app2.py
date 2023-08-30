### Imports
import streamlit as st
import os
import io
import openai
from datetime import datetime
import pandas as pd
import mysql.connector as database
import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path="langchain.db")
from langchain.agents import initialize_agent, AgentType
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, Tool, tool
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.llms import OpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain import LLMMathChain
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import create_python_agent
from contextlib import redirect_stdout
from typing import Optional, Type
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

def run_inquiry(myquestion):
    st.session_state.messages.append({"role": "user", "content": myquestion})
    st.chat_message("user").write(myquestion)
    with st.spinner("Please wait..."):

        docs = db3.similarity_search_with_relevance_scores(myquestion, k=8)

        mytasks = ""
        for x,v in docs:
            if mytasks == "":
                mytasks = str(x) + "\n\n"
                continue
            if v > relevancy_cutoff:
                mytasks = mytasks + str(x) + "\n\n"
        # print(mytasks)

        template=f"""You are a helpful chatbot trying to answer the user's QUESTION below. 
        Using the TEXT below, develop a series of steps that could lead to an answer.
            
            TEXT:
            {mytasks}

            QUESTION:
            {myquestion}
            Provide a justification for your answer.
            """
        # print(template)

        response = openai.ChatCompletion.create(
            messages = [{"role": "user", "content": template}],
            temperature=0,
            max_tokens=2000,
            frequency_penalty=0,
            presence_penalty=0,
            model="gpt-4")

        newprompt = response.choices[0].message.content
        # print(newprompt)

        response = search_agent(newprompt)
        final_prompt = response['output']
        # print(response['output'])

        final_template=f"""You are a helpful chatbot trying to answer the user's QUESTION below. 
        Using the TEXT below, write a final answer to the user's question.
            
            TEXT:
            {final_prompt}

            QUESTION:
            {myquestion}
            Provide all the details available in your answer.
            """
        # print(final_template)

        response = openai.ChatCompletion.create(
            messages = [{"role": "user", "content": final_template}],
            temperature=0,
            max_tokens=2000,
            frequency_penalty=0,
            presence_penalty=0,
            model="gpt-4")

        answer = response.choices[0].message.content

        st.session_state.messages.append({"role": "assistant", "content": answer})    
        st.chat_message('assistant').write(answer)

        if show_detail:
            with st.expander('Details', expanded=False):
                s = f.getvalue()
                st.write(s)
        return

### Create SQL Chain
class MySQLTool(BaseTool):
    name = "MySQLTool"
    description = """
    This tool is used for running SQL queries against a MariaDB database. 
    It is useful for when you need to answer questions about employee access 
    activities by running MariaDB queries. Always end your SQL queries with a ";". 

    The following table information is provided to help you write your sql statement. 
    The "demo_access_log" table is a timestamp file of employee activities. The employee is identified by the badge number. 
    The event types in the demo_access_log table include: 
       "BE" = "Building Entry"
       "BX" = "Building Exit"
       "RE" = "Room Entry"
       "RX" = "Room Exit"
    You should not see a "BX" entry type without a "BE" occuring first. 
    You should not see a "RX' entry type without a "RE" occuring first.
    The "demo_emp" table translates tthe badge number to an employee name and assigned department. This table can be joined to the demo_access_log table using the "badge" column
    The "demo_site" table translates the site number to a city and state for that site. This table can be joined to the demo_access_log table using the "site" column.
    The "demo_room" table translates the room at a particular site into a room name. This table can be joined to the demo_access_log table using the "site" and "room" columns.
            
    demo_access_log: (timestamp, badge, event, site, room)
    demo_emp: (badge, name, department)
    demo_site: (site, city, state)
    demo_room: (site, room, name)    
    """
    
    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""

        prefix1 = "```sql"
        prefix2 = "``````"
        prefix3 = "```"
        
        # st.write("AAA: "+query+" :AAA")
        if query.startswith(prefix1):
            newquery = query[7:-4]
        elif query.startswith(prefix2):
            newquery = query[7:-4]
        elif query.startswith(prefix3):
            newquery = query[4:-4]
        else: 
            newquery = query
 
        try:
            # st.write("BBB: "+newquery+" :BBB")
            mycursor.execute(newquery)
            results = mycursor.fetchall()
        except database.Error as e:
            results = "Error running query"
        
        return results  
                
    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")

llm = ChatOpenAI(model="gpt-4", temperature=0)

tools = [MySQLTool()]

search_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True,
    return_intermediate_steps=True,
)

### Create Vectorstore
raw_documents = TextLoader('content/runbook.txt').load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db3 = Chroma.from_documents(documents, OpenAIEmbeddings())
relevancy_cutoff = .7
# myprompts = pd.read_csv('content/myprompts.csv')

### Create SQL Chain
myusername = "streamlit"
mypassword = "streamlitpass"

myconnection = database.connect(
    user=myusername,
    password=mypassword,
    host="10.1.0.4",
    database="streamlit")

mycursor = myconnection.cursor()

### OpenAI Stuff
# OpenAI Credentials
if not os.environ["OPENAI_API_KEY"]:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai_api_key = os.environ["OPENAI_API_KEY"]

### Streamlit Stuff
### CSS
st.set_page_config(
    page_title='GAMEPLAN', 
    layout="wide",
    initial_sidebar_state='collapsed',
)
padding_top = 0
st.markdown(f"""
    <style>
        .block-container, .main {{
            padding-top: {padding_top}rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)
### UI
col1, col2 = st.columns( [1,5] )
col1.write("")
col1.image('GTLogo.png', width=170)
col2.write("")
col2.subheader('GT AI Driven Cyber Analytics Platform')

with st.sidebar: 
    mysidebar = st.selectbox('Select Model', ['Employee Access'])
    if mysidebar == 'Employee Access':
        show_detail = st.checkbox('Show Details')
        llm_model = st.selectbox('Select Model', ['gpt-4', 'gpt-3.5-turbo'])
        st.markdown("---")
        st.markdown("### Standard Questions:")
        bt1 = st.button('Still Inside')
        bt2 = st.button('What about Bob?')
        st.markdown("---")
        tz = st.container()

if mysidebar == 'Employee Access':
    with st.expander("**:blue[Employee Access Overvew]**"):
        st.markdown("**:blue[This LLM chain uses a Planner/Executor SuperChain.]**")
        st.markdown("**:blue[The workflow first enters the Planner phase where it uses vector semantic KNN search of our \"run book\" document to determine how to answer the question. This document resembles an FAQ document and provides steps for completing these tasks. These are tasks a human might take. The LLM will translate these steps into steps that LangChain can execute.]**")
        st.markdown("**:blue[In the Executor phase the LLM instructs LangChain how to perform each step. Answers from each step are preserved and a final answer is generated by the LLM to the proposed questions. For this demo LLM largely relies on SQL queries. However, the workflow could include other operations including dynamic Python using SciKit, querying the internet, running shell scripts, running REST queries, or any act that might be defined in the CSIO's document. Chains are toolsets that must be assembled to meet a general area of inquiry.]**")

        col1, col2, col3 = st.columns([15, 70, 15])
        col2.image('cyber.jpg',caption='LangChain Structure')

    st.markdown('---')

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="Ask a question?"):
        start = datetime.now()
        tz.write("Start: "+str(start)[10:])
        if show_detail:
            f = io.StringIO()
            with redirect_stdout(f):
                run_inquiry(prompt)
        else:
            run_inquiry(prompt)
        tz.write("End: "+str(datetime.now())[10:])
        tz.write("Duration: "+str(datetime.now() - start))
    if bt1:
        start = datetime.now()
        tz.write("Start: "+str(start)[10:])
        if show_detail:
            f = io.StringIO()
            with redirect_stdout(f):
                run_inquiry("Is Bob still inside the building?")
        else:
            run_inquiry("Is Bob still inside the building?")
        tz.write("End: "+str(datetime.now())[10:])
        tz.write("Duration: "+str(datetime.now() - start))
    if bt2:
        start = datetime.now()
        tz.write("Start: "+str(start)[10:])
        if show_detail:
            f = io.StringIO()
            with redirect_stdout(f):
                run_inquiry("Is Bob exhibiting unusual behavior?") 
        else:
            run_inquiry("Is Bob exhibiting unusual behavior?") 
        tz.write("End: "+str(datetime.now())[10:])
        tz.write("Duration: "+str(datetime.now() - start))
