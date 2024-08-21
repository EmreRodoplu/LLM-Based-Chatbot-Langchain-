from langchain_openai import ChatOpenAI,OpenAIEmbeddings 
from langchain_community.vectorstores.faiss import FAISS 
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts.prompt import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnablePassthrough 
from flask import Flask,request
from dotenv import load_dotenv 
import os
from waitress import serve

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")



#AI MODEL
llm = ChatOpenAI(model="gpt-4o-mini")

#EMBEDDING MODEL
embedding = OpenAIEmbeddings()

#SPLITTER
loader = PyPDFLoader(file_path="machine_learning.pdf")
document = [pages.page_content for pages in loader.load_and_split()]

template = PromptTemplate.from_template("""
Answer the question based on the following context:    
{document}

Detect the language of given question and response back in the same language
if the given question has nothing to do with the context you will politely response back that you can not help
it will be short answer
    
Question : {query}
                                        """)

#CONNECT TO VECTOR DATABASE
vectordb = FAISS.from_texts(texts=document,embedding=embedding)

retriver = vectordb.as_retriever()

chain= ({"document":retriver,"query":RunnablePassthrough()} | template | llm | StrOutputParser())

app = Flask(__name__)

@app.route("/Chatbot",methods=["POST"])
def ChatBot():
    
    json_content = request.json
    query= json_content["query"]

    return {"answer":chain.invoke(query)}
    
if __name__ == "__main__":
    # serve(app = app, host ="localhost", port = 5000)
    app.run(host="localhost",port=5000,debug=True)