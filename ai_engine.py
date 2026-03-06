import os
from dotenv import load_dotenv

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from rag import store_embeddings, retriever

from pydantic import BaseModel
from typing import List
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# Initialize LLM
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash", api_key=os.environ.get("API_KEY"), temperature=0.3
# )

llm = ChatMistralAI(api_key=os.environ.get("MISTRAL_API_KEY"), temperature=0.3)


# Load PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    document = loader.load()
    return document


# Split into chunks
def split_text(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    texts = text_splitter.split_documents(document)
    return texts


# 🔹 Summary Prompt for Each Chunk
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
        You are an expert academic study assistant.

        Summarize the following section of study notes clearly and concisely.

        Instructions:
        - Use bullet points
        - Preserve important concepts and terminology
        - Include definitions if present
        - Do NOT remove key technical details
        - Keep summary structured and easy for revision

        TEXT:
        {text}

        SUMMARY:
        """,
)


# Summarize all chunks


def generate_chunk_summaries(texts):

    def summarize_chunk(text, i):
        print(f"Summarizing chunk {i+1}/{len(texts)}...")

        formatted_prompt = summary_prompt.format(text=text)

        response = llm.invoke(formatted_prompt)

        return response.content

    max_workers = 5

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunk_summaries = list(
            executor.map(
                lambda args: summarize_chunk(*args),
                [(text, i) for i, text in enumerate(texts)],
            )
        )

    return chunk_summaries


def generate_complete_summary(complete_summary):
    complete_summary_prompt = PromptTemplate(
        template="""You are an expert academic assistant.

            Create a final structured revision summary.

            Instructions:
            - Use clear headings and bullet points
            - Preserve important concepts and terminology
            - Include definitions if present
            - Keep summary structured and easy for revision

            OUTPUT RULES (VERY IMPORTANT):
            - Return ONLY pure HTML.
            - **DO NOT wrap the response in ```html or markdown code blocks.**
            - **DO NOT include backticks (`).**
            - **The output must be directly renderable HTML.**

            STYLING RULES:
            - The application uses a DARK background.
            - DO NOT use white backgrounds.
            - Use light text colors suitable for dark mode.
            - Prefer colors like:
            - text color: #e5e7eb or white
            - headings: #a5b4fc or #c7d2fe
            - Avoid inline backgrounds unless dark-compatible.

            Content:
            {combined_text}
        """,
        input_variables=["combined_text"],
    )

    formatted_prompt = complete_summary_prompt.format(combined_text=complete_summary)
    response = llm.invoke(formatted_prompt)

    return response


def generate_questions(complete_summary):
    class QuestionItem(BaseModel):
        question: str
        answer: str
        explanation: str

    class QuestionList(BaseModel):
        questions: List[QuestionItem]

    parser = PydanticOutputParser(pydantic_object=QuestionList)

    questions_prompt = PromptTemplate(
        template="""You are an expert academic examiner.

            Generate 20 exam-oriented questions from the study material.

            Instructions:
            - 5 Short Answer Questions
            - 5 Long Answer Questions
            - 5 Conceptual Questions
            - 5 Application-Based Questions
            - Each question must include:
                - question
                - answer
                - explanation
            - Cover all major topics
            - Avoid repetition

            STUDY MATERIAL:
            {text}

            Return ONLY valid JSON.
            Do NOT include explanations, markdown, or extra text. In below format:
            {format_instructions}
            """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = questions_prompt | llm | parser
    response = chain.invoke({"text": complete_summary})

    return response.model_dump()


def generate_flashcards(complete_summary):
    class FlashCardItem(BaseModel):
        id: int
        front: str
        back: str
        difficulty: str
        topic: str

    class FlashCardList(BaseModel):
        flashcards: List[FlashCardItem]

    parser = PydanticOutputParser(pydantic_object=FlashCardList)

    flashcards_prompt = PromptTemplate(
        template="""You are an expert academic study assistant.

            Create high-quality revision flashcards from the study material.

            Instructions:
            - Generate 15–25 flashcards
            - Each flashcard must:
                - Have a short clear question on the front
                - Have a concise but complete answer on the back
                - Include difficulty level: easy / medium / hard
                - Include topic name
            - Keep answers under 5 lines
            - Avoid overly long paragraphs
            - Make them exam-focused

            STUDY MATERIAL:
            {text}

            Return ONLY valid JSON.
            Do NOT include explanations, markdown, or extra text. In below format:
            {format_instructions}""",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = flashcards_prompt | llm | parser
    response = chain.invoke({"text": complete_summary})

    return response.model_dump()


def generate_quiz(complete_summary):
    class QuizItem(BaseModel):
        id: int
        question: str
        options: List[str]
        correct_answer: str
        explanation: str
        difficulty: str

    class QuizList(BaseModel):
        quizzes: List[QuizItem]

    parser = PydanticOutputParser(pydantic_object=QuizList)

    quizzle_prompt = PromptTemplate(
        template="""You are an expert academic examiner.

            Generate a multiple-choice quiz from the study material.

            Instructions:
            - Generate 15 questions
            - Each question must:
                - Have 4 options
                - Only ONE correct answer
                - Include explanation
                - Include difficulty level (easy/medium/hard)
            - Avoid repetition
            - Cover all major topics
            - Make questions exam-oriented

            STUDY MATERIAL:
            {text}

            Return ONLY valid JSON.
            Do NOT include explanations, markdown, or extra text. In below format:
            {format_instructions}""",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = quizzle_prompt | llm | parser
    response = chain.invoke({"text": complete_summary})

    return response.model_dump()


def ask_question(question, session_id):
    docs = retriever.retrieve_docs(question, k=3, session_id=session_id)
    context = retriever.format_context(docs)

    prompt = PromptTemplate(
        template="""You are an assistant answering questions using the provided context.

            Context:
            {context}

            Question:
            {question}

            Answer using ONLY the context above.
            If answer is not found, say "Not in documents".
            """,
        input_variables=["context", "question"],
    )

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    return {"answer": response.content}


# Run pipeline
# document = load_pdf("data.pdf")
# texts = split_text(document)

# result = ask_question("what are the submission requirements?", "123", texts)
# print(result)

# chunk_summaries = generate_chunk_summaries(texts)
# complete_summary = "\n".join(chunk_summaries)

# summary = (generate_complete_summary(complete_summary)).content
# print("""Summary:::::::::::::::::::::::::::::::::::::::::::::::""", summary)

# questions = generate_questions(summary)
# print("""Questions:::::::::::::::::::::::::::::::::::::::::::""", questions)

# flashcards = generate_flashcards(summary)
# print("""Flashcards::::::::::::::::::::::::::::::::::::::::::""", flashcards)

# quiz = generate_quiz(summary)
# print("""Quiz:::::::::::::::::::::::::::::::::::::::::::""", quiz)
