import bs4
import requests
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_URLS = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]


def load_web_page(url: str, bs_kwargs: dict | None = None) -> list[Document]:
    """Fetch a web page and wrap its text content as a single Document."""
    response = requests.get(url)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, "html.parser", **(bs_kwargs or {}))
    return [Document(page_content=soup.get_text(), metadata={"source": url})]


def build_retriever(
    urls: list[str] = DEFAULT_URLS,
    chunk_size: int = 100,
    chunk_overlap: int = 50,
):
    """Load `urls`, chunk them, and return a retriever over an in-memory vector store."""
    docs = [doc for url in urls for doc in load_web_page(url)]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    doc_splits = text_splitter.split_documents(docs)

    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits, embedding=OpenAIEmbeddings()
    )
    return vectorstore.as_retriever()
