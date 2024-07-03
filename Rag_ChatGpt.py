from langchain_community.llms.ollama import Ollama 
from langchain_openai import OpenAI,OpenAIEmbeddings 
from langchain_community.vectorstores.faiss import FAISS 
from langchain_community.embeddings.ollama import OllamaEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import TextLoader , PyPDFLoader
from langchain.chains import LLMChain,ConversationalRetrievalChain,create_retrieval_chain,RetrievalQA 
from langchain_core.prompts.prompt import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnablePassthrough 
from flask import Flask,request
from deepl import Translator 
from dotenv import load_dotenv 
import os
from waitress import serve


#API KEY FOR DEEPL
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
deepl_key = os.getenv("DEEPL")


#AI MODEL
llm = OpenAI()

#EMBEDDING MODEL
embedding = OpenAIEmbeddings()

#SPLITTER
loader = PyPDFLoader(file_path="machine_learning.pdf")
document = [pages.page_content for pages in loader.load_and_split()]

template = PromptTemplate.from_template("""
AS an AI asisstant named ROBEL Answer the question based only on the following context:    
{document}

Start a sentence 'According to the ROBEL'
if the given question has nothing to do with the context you will politely response back that you can not help
it will be short answer
    
Question : {query}
                                        """)

#CONNECT TO VECTOR DATABASE
vectordb = FAISS.from_texts(texts=document,embedding=embedding)

retriver = vectordb.as_retriever()

chain= ({"document":retriver,"query":RunnablePassthrough()} | template | llm | StrOutputParser())

# chain = RetrievalQA.from_chain_type(llm=llm,retriever=vectordb.as_retriever())
# chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vectordb.as_retriever())
# chain = RetrievalQA.from_llm(llm=llm,retriever=vectordb.as_retriever())

translator = Translator(deepl_key)

app = Flask(__name__)

@app.route("/Chatbot/<query>",methods=["GET"])
def ChatBot(query):
    #Translate any given sentence to English 
    translate_to_English = translator.translate_text(query, target_lang="EN-US")
    #IF given language is English
    if translate_to_English.detected_source_lang == "EN":
        return {"answer":chain.invoke(query)}
    else:
        #Translate LLM response back to detected source language
        result = translator.translate_text(chain.invoke(translate_to_English.text),target_lang=translate_to_English.detected_source_lang)
        return {"answer":result.text}
    
if __name__ == "__main__":
    # serve(app = app, host ="localhost", port = 5000)
    app.run(host="localhost",port=5000,debug=True)