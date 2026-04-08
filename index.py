import streamlit as st
from backend import workflow
from langchain_core.messages import HumanMessage

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

for message in st.session_state['message_history']:
    with st.chat_message(message["role"]):
        st.text(message["content"])
  
    
user_input = st.chat_input("Ask me anything about Streamlit!")
if user_input:
    
    st.session_state['message_history'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)
    
    config = {"configurable": {"thread_id": "bollywood_chat_thread"}}
    response = workflow.invoke({"messages": HumanMessage(content=user_input)}, config=config)
  
    st.session_state['message_history'].append({"role": "assistant", "content": response['messages'][-1].content})
    with st.chat_message("assistant"):
        st.text(response['messages'][-1].content)    