from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import GRADER_MODEL_DEPLOYMENT, OPENAI_API_KEY
from .documents import build_retriever

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            (
                "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
                "Here is the initial question:"
                "\n ------- \n"
                "{question}"
                "\n ------- \n"
                "Formulate an improved question:"
            ),
        )
    ]
)

_retriever = None


def get_retriever():
    """Build (once) and return the blog post retriever."""
    global _retriever
    if _retriever is None:
        _retriever = build_retriever()
    return _retriever


def retrieve_blog_posts(query: str) -> dict:
    """Stateless: Search and return information about Lilian Weng blog posts."""
    print("---TOOLCALL: RETRIEVING CONTEXT---")
    docs = get_retriever().invoke(query, top_k=5)
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"context": context, "source": "Lilian Weng blog posts Knowledge Base"}


def rewrite_question(query: str) -> dict:
    """Stateless: Rewrite the original user question."""
    print("---TOOLCALL: REWRITE QUESTION---")
    response_llm = ChatOpenAI(model=GRADER_MODEL_DEPLOYMENT, temperature=0, api_key=OPENAI_API_KEY)
    chain = REWRITE_PROMPT | response_llm
    response = chain.invoke({"question": query})
    return {"context": response.content, "source": "rewrite_question"}


TOOL_MAPPING_2_FUNCTIONS = {
    "retrieve_blog_posts": retrieve_blog_posts,
    "rewrite_question": rewrite_question,
}

AGENT_TOOL_LIST = {
    "agent_main_generate_query_or_respond": [
        {
            "name": "retrieve_blog_posts",
            "description": "Retrieve relevant information about Lilian Weng blog posts in knowledge base.",
            "args": "query (string)",
        }
    ],
    "agent_rewrite_question": [
        {
            "name": "rewrite_question",
            "description": "Rewrite the original user question.",
            "args": "query (string)",
        }
    ],
}
