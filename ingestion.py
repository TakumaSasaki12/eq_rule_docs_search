import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
# from langchain_pinecone import Pinecone
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from langchain_pinecone import PineconeVectorStore
from langchain_community.chat_models import ChatOpenAI
from consts import INDEX_NAME, R6_RULE_URL



def extract_text_from_pdf(pdf_path):
    # PDFファイルを開いて読み込みます
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        # ページ数を取得します
        num_pages = len(reader.pages)
        # 各ページのテキストを抽出します
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text = page.extract_text()
            # テキストをファイルに保存します
            text_path = f"output/page_{page_num + 1}.txt"
            with open(text_path, "w", encoding="utf-8") as text_file:
                text_file.write(text)

            url = R6_RULE_URL + f"#page={page_num}"
            docs = sep_docs(text_path, page_num, url)
            print(docs)

            try:
                os.remove(text_path)
                print(f"ファイル {text_path} を削除しました。")
            except OSError as e:
                print(f"ファイル {text_path} を削除できませんでした: {e}")

            embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

            index = pc.Index(index_name)
            index.describe_index_stats()

            docsearch = PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
 

def sep_docs(text_path, page_num, url):
    loader = TextLoader(text_path, encoding="utf-8")
    documents = loader.load()
    print(len(documents))
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)
    for doc in docs:
        doc.metadata["page_num"] = page_num
        doc.metadata["url"] = url
        doc.metadata["name"] = "競技会関連規程集（令和6年度版）"
    return docs


if __name__ == "__main__":

    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

    # インデックス名を適切に設定する
    index_name = INDEX_NAME

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536, # Replace with your model dimensions
            metric="euclidean", # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )

    model_name = 'text-embedding-ada-002'
    embeddings = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.environ.get("OPENAI_API_KEY")    
    )
    text_field = "text"
    
    index = pc.Index(index_name)
    print(index.describe_index_stats())

    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )

    from langchain.vectorstores import Pinecone
    text_field = "text"

    # switch back to normal index for langchain
    index = pc.Index(index_name)

    vectorstore = Pinecone(
        index, embed.embed_query, text_field
    )

    query = """
    基準Aについて説明してください。
    """

    print(
        vectorstore.similarity_search(
            query,  # our search query
            k=3  # return 3 most relevant docs
        )
    )
    

    from langchain_community.chat_models import ChatOpenAI
    from langchain.chains.conversation.memory import ConversationBufferWindowMemory
    from langchain.chains import RetrievalQA

    # chat completion llm
    llm = ChatOpenAI(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        model_name='gpt-4',
        temperature=0.0
    )
    # conversational memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )
    # retrieval qa chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    result = qa.invoke({"query":query})
    print(result)

    # from langchain.agents import Tool

    # tools = [
    #     Tool(
    #         name='Knowledge Base',
    #         func=qa.run,
    #         description=(
    #             "useful for when you need to search for latest information in web"
    #             "please anser by Japanese"
    #         )
    #     )
    # ]

    # from langchain.agents import initialize_agent
    # from langchain.agents import AgentType

    # agent = initialize_agent(
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     tools=tools,
    #     llm=llm,
    #     verbose=True,
    #     max_iterations=10,
    #     early_stopping_method='generate',
    #     memory=conversational_memory
    # )

    # print(agent(query))