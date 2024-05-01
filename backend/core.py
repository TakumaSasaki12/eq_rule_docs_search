# from dotenv import load_dotenv

# load_dotenv()

import os
from typing import Any, List, Dict

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Pinecone
import pinecone

from consts import INDEX_NAME

# pinecone.init(
#     api_key=os.environ["PINECONE_API_KEY"],
#     environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
# )

# INDEX_NAME = "langchain-doc-index"


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Any:
    model_name = 'text-embedding-ada-002'
    embeddings = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.environ.get("OPENAI_API_KEY")    
    )

    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    
    chat = ChatOpenAI(verbose=True, temperature=0)

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, 
        retriever=docsearch.as_retriever(), 
        return_source_documents=True
    )

    return qa.invoke({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="公認審判長に必要な資格は?"))
