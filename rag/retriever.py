from langchain_community.vectorstores import Chroma
from .embedding import get_embedding_model


def get_vectorstore(session_id):
    vectorstore = Chroma(
        persist_directory="db/" + session_id,
        embedding_function=get_embedding_model(),
    )

    return vectorstore


def retrieve_docs(question, k=3, session_id="default"):
    vectoreStore = get_vectorstore(session_id)

    docs = vectoreStore.similarity_search(question, k=k)

    return docs


def format_context(docs):
    return "\n\n".join([doc.page_content for doc in docs])
