from langchain_community.vectorstores import Chroma
from .embedding import get_embedding_model


def store_embeddings(chunks, session_id):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        persist_directory="db/" + session_id,
    )

    vectorstore.persist()
