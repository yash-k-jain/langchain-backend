import os
import uuid
from langchain_core.documents import Document
from flask import Flask, request, jsonify
from ai_engine import (
    load_pdf,
    split_text,
    generate_chunk_summaries,
    generate_complete_summary,
    generate_questions,
    generate_quiz,
    generate_flashcards,
)
from redis_service import save_session, get_session, update_session_field
from flask_cors import CORS
from ai_engine import ask_question
from rag import store_embeddings

app = Flask(__name__)

CORS(app, origins=["http://localhost:5173"])


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/upload", methods=["POST"])
def upload_pdf():
    if not request.files.get("file"):
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    session_id = str(uuid.uuid4())

    file_path = os.path.join(UPLOAD_FOLDER, f"{session_id}.pdf")
    file.save(file_path)

    document = load_pdf(file_path)
    texts = split_text(document)
    store_embeddings.store_embeddings(texts, session_id)

    # ✅ store chunks per user session
    save_session(
        session_id,
        {
            "texts": [doc.page_content for doc in texts],
            "summary": None,
            "questions": None,
            "flashcards": None,
            "quiz": None,
        },
    )

    os.remove(file_path)

    return jsonify({"message": "Upload successful", "session_id": session_id})


@app.route("/summary", methods=["POST"])
def summary():

    session_id = request.json.get("session_id")
    session = get_session(session_id)

    if not session:
        return jsonify({"error": "Invalid session"}), 400

    if session.get("summary"):
        return jsonify({"summary": session.get("summary")})

    texts = session.get("texts")

    chunk_summaries = generate_chunk_summaries(texts)
    combined = "\n".join(chunk_summaries)

    final_summary = generate_complete_summary(combined).content

    update_session_field(session_id, "summary", final_summary)

    return jsonify({"summary": final_summary})


@app.route("/questions", methods=["POST"])
def questions():

    session_id = request.json.get("session_id")
    session = get_session(session_id)

    if not session:
        return jsonify({"error": "Invalid session"}), 400

    if session.get("questions"):
        return jsonify({"questions": session.get("questions")})

    summary = session.get("summary")

    questions = generate_questions(summary)

    update_session_field(session_id, "questions", questions)

    return jsonify({"questions": questions})


@app.route("/flashcards", methods=["POST"])
def flashcards():

    session_id = request.json.get("session_id")
    session = get_session(session_id)

    if not session:
        return jsonify({"error": "Invalid session"}), 400

    # Return cached result if exists
    if session.get("flashcards"):
        return jsonify({"flashcards": session.get("flashcards")})

    summary = session.get("summary")

    flashcards = generate_flashcards(summary)

    update_session_field(session_id, "flashcards", flashcards)

    return jsonify({"flashcards": flashcards})


@app.route("/quiz", methods=["POST"])
def quiz():

    session_id = request.json.get("session_id")
    session = get_session(session_id)

    if not session:
        return jsonify({"error": "Invalid session"}), 400

    if session.get("quiz"):
        return jsonify({"quiz": session.get("quiz")})

    summary = session.get("summary")

    quiz = generate_quiz(summary)

    update_session_field(session_id, "quiz", quiz)

    return jsonify({"quiz": quiz})


@app.route("/qa", methods=["POST"])
def qa():
    session_id = request.json.get("session_id")
    question = request.json.get("question")

    session = get_session(session_id)
    if not session:
        return jsonify({"error": "Invalid session"}), 400

    result = ask_question(question, session_id)
    return jsonify({"result": result})


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
