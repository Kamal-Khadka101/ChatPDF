import os
import warnings
from dotenv import load_dotenv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

load_dotenv()

pdfs = []

for root, dirs, files in os.walk('rag-dataset'):
    #print(root, dirs, files)
    for file in files:
        if file.endswith('.pdf'):
            pdfs.append(os.path.join(root, file))

from langchain_community.document_loaders import PyMuPDFLoader

docs = []
for pdf in pdfs:
    loader = PyMuPDFLoader(pdf)
    pages = loader.load()
    docs.extend(pages)

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap = 80)

chunks = text_splitter.split_documents(docs)


import tiktoken

encoding = tiktoken.encoding_for_model('gpt-4o-mini')

len(encoding.encode(docs[0].page_content)), len(encoding.encode(chunks[0].page_content))


from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")

test_vector = embeddings.embed_query("I believe this query will also generate a fixed number of vectors which will define the dimensions")

len(test_vector)

index = faiss.IndexFlatL2(len(test_vector))
index.ntotal


vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

ids = vector_store.add_documents(documents=chunks)



question = "What is used to gain muscle mass?"
vector_store.search(query=question, search_type='similarity')

similar_chunks = vector_store.search(query=question, search_type='similarity')

for parts in similar_chunks:
    print(parts.page_content)
    print("\n\n")

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs = {'k': 3, 'fetch_k': 100, 'lambda_mult': 1})

similar_chunks = retriever.invoke(question)

for parts in similar_chunks:
    print(parts.page_content)
    print("\n\n")

question = "what is used to reduce weight?"
# question = "what are side effects of supplements?"
# question = "what are the benefits of supplements?"
# question = "what are the benefits of BCAA supplements?"
try_data = retriever.invoke(question)



# %%
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from langchain_ollama import ChatOllama

# %%
model = ChatOllama(model="llama3.2:3b", base_url="http://localhost:11434")

prompt = hub.pull("rlm/rag-prompt")



def format_docs(similar_chunks):
    return "\n\n".join([parts.page_content for parts in similar_chunks])

# print(format_docs(similar_chunks))

rag_chain = (
    {"context": retriever|format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# %%
# question = "what is used to gain muscle mass?"
# question = "what is used to reduce weight?"
# question = "what are side effects of supplements?"
# question = "what are the benefits of supplements?"
# question = "what are the benefits of BCAA supplements?"

question = "what is used to increase mass of the Earth?"

output = rag_chain.invoke(question)
print(output)




