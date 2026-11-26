import streamlit as st
from langchain_ollama import ChatOllama
from langchain_classic.chains import LLMChain,LLMMathChain
from langchain_classic.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents import Tool,initialize_agent
from langchain_classic.callbacks import StreamlitCallbackHandler

# Set up the streamlit app
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Asisstant")
st.title("Text to Math Problem Solver")

llm=ChatOllama(model="gemma2:latest")

## Initializing the tools
wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the internet to find the various inforamtion on the topics mentioned"
)

## Initialize the Math tool
math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="calculator",
    func=math_chain.run,
    description="A tool for answering math realted wuestion. only input mathematical expression need to be provided"
)

prompt="""
You are a agent tasked for solving users mathemtical questions. Logically arrive at the solution
and display it point wise for the question below
Question:{question}
Answer:
"""
prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

## Combine all the tools into chain
chain=LLMChain(llm=llm,prompt=prompt_template)

resoning_tool=Tool(
    name="reasoning_tool",
    func=chain.run,
    description="A tool for answering logic-based and resoning questions."
)
## initialize the agents 
asisstant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,resoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_error=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"asisstant","content":"Hi, I am a math chatbot who can answer all your maths question"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])


## Lets start the interaction
question=st.text_area("Enter your question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

if st.button("find my answer"):
    if question:
        with st.spinner("Generate response.."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write()
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=asisstant_agent.run(st.session_state.messages,callbacks=[st_cb])
            st.session_state.messages.append({'role':'asisstant','content':response})
            st.write('### Response: ')
            st.success(response)
    else:
        st.warning("Please enter ther question")