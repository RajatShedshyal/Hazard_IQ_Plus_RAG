#HOW TO RUN ->
# 1. host this file using -> python -m fastapi run netra_api.py --port 8001
# 2. use Cloudflare Tunnel to expose the port use -> cloudflared tunnel --url http://localhost:8001

import os
from dotenv import load_dotenv

from fastapi import FastAPI
from pydantic import BaseModel

######################------------ CHAT BOT API -------------######################

#Imports for Pinecone and Vector Store who help u store the Docs on vector database
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding 

#Imports for Query Engine and again for Semantic Search using Vector_Store_Index
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine


from llama_index.llms.gemini import Gemini #LLM

from llama_index.core import Settings

load_dotenv()  # Load environment variables from .env file

llm = Gemini()
embed_model = GeminiEmbedding(model_name="models/embedding-001")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 1024 #smaller value is more precise that is after every 1024 characters it will create a new chunk

pinecone_client = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))


pinecone_index = pinecone_client.Index("chatbot6")  # The Index name
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
retriever = VectorIndexRetriever(index=index, similarity_top_k=5)

#Lets BUILD one Contextual Query Engine
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine.types import ChatMessage

######################------------ SQL SHIT -------------######################
import psycopg2

DB_HOST = os.environ.get("DB_HOST")
DB_DB = os.environ.get("DB_DB")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_PORT = os.environ.get("DB_PORT")

curr = None
conn = None

try:

  conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_DB,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

  curr = conn.cursor()

except Exception as e:
  print(f"Error connecting to the database: {e}")
  raise e

######################------------  JUST API  ------------######################

app = FastAPI()

class User_Id(BaseModel):
    user_id: str

@app.get("/")
def default():
    return(f"end point is '/query' ")

@app.post("/query")
def get_query_from_user(user: User_Id):
    id = user.user_id
    print(f"Id passed: {id}")

    #chat_history for stroing context ->
    chat_history = []

    #lets fetch all the required stuffs ->

    # Execute the query

    try:

        create_table = """ 
            CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            role VARCHAR(10) NOT NULL,
            message TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
         ); """
        
        curr.execute(create_table)

        curr.execute("""
            SELECT *
            FROM chat_history
            WHERE user_id = %s
            ORDER BY timestamp
        """, (id,))

        chat_history = []

        # Fetch all matching rows
        rows = curr.fetchall()

        for row in rows:
            chat_history.append({
                "role": row[2],
                "content": row[3],
            })

        print("### Chat History Fetched Successfully ###")

        user_question = chat_history[-1]["content"] if chat_history else "you respond with How can i help you today?"

        memory = ChatMemoryBuffer.from_defaults()

        for msg in chat_history:
            msg = ChatMessage(role=msg['role'], content=msg["content"])
            memory.put(msg)

        #Contextual Chat Engine

        chat_engine = ContextChatEngine.from_defaults(
            retriever=retriever,
            llm=llm,
            memory=memory
        )

        # Single Convo

        import time
        start_time = time.time()
        response = chat_engine.chat(user_question)
        end_time = time.time()

        # put back the response to the database

        insert_data = """
          INSERT INTO chat_history (user_id, role, message)
          VALUES (%s, %s, %s)
        """

        answer = response.response.strip()
        curr.execute(insert_data, ("12345", "assistant", response.response.strip()))

        conn.commit()

        print(f"Time taken: {end_time - start_time} secs,\n Bot: {answer}")
        return {
            "response": answer
        }

        ## reput the chat history back to the database

    except Exception as e:
        print(f"Error fetching data from the database: {e}")
        return {"error": "Failed to fetch data from the database, maybe wrong user_id?"}



