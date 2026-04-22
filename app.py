from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-nano",api_key=os.getenv("OPENAI_API_KEY"))

# resp1 = llm.invoke("We are building an AI system for processing medical insurance claims.")
# print(resp1.content)
# print("-------------------------------------------------")
# resp2 = llm.invoke("What are the main risks in this system?")
# print(resp2.content)

"""
Why the second question may fail or behave inconsistently without conversation history.
The second question may fail or behave inconsistently without conversation history because the model does not have the context of the first question and its response. The model may not understand that the second question is related to the first one, and it may not be able to provide a relevant answer. Additionally, without conversation history, the model may not be able to maintain a coherent thread of discussion, leading to responses that are disconnected or irrelevant to the previous interactions.
"""

messages = [
    SystemMessage(content="You are a senior AI architect reviewing production systems."),
    HumanMessage(content="We are building an AI system for processing medical insurance claims."),
    HumanMessage(content="What are the main risks in this system?")
]
resp3 = llm.invoke(messages)
print(resp3.content)

"""
Reflection:

1. Why did string-based invocation fail?
String based invocation failed because the model did not have the necessary context from the previous interaction. The model treated each input as an isolated query, and without the conversation history, it could not connect the second question to the first one. This lack of context led to a failure in providing a relevant and coherent response to the second question.
2. Why does message-based invocation work?
Message-based invocation works because it provides the model with the entire conversation history, allowing it to understand the context and maintain a coherent thread of discussion. The model can reference previous messages, which helps it to generate responses that are relevant and consistent with the ongoing conversation. This approach enables the model to better understand the user's intent and provide more accurate and meaningful answers.
3. What would break in a production AI system if we ignore message history?
Ignoring message history in a production AI system would lead to several issues. The system would struggle to maintain context across interactions, resulting in responses that are disconnected and irrelevant to the user's previous inputs. This could lead to user frustration and a poor user experience, as the AI would fail to provide coherent and meaningful responses. Additionally, without message history, the AI would be unable to learn from past interactions, which could hinder its ability to improve over time and adapt to user needs effectively.
"""