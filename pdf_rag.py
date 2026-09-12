import os
import shutil
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM


# ============================================================
# 1. SETTINGS
# ============================================================

PDF_FOLDER = "uploaded_pdf"
VECTORSTORE_FOLDER = "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"


# Number of chunks retrieved from FAISS
RETRIEVAL_K = 8

# Number of chunks used for searching before MMR selection
FETCH_K = 25


# ============================================================
# 2. FIND PDF
# ============================================================

def find_pdf():

    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)

    pdf_files = [
        file
        for file in os.listdir(PDF_FOLDER)
        if file.lower().endswith(".pdf")
    ]

    if len(pdf_files) == 0:

        print("\n❌ No PDF found.")

        print(
            "\nPlease put your PDF inside:"
        )

        print(
            os.path.abspath(PDF_FOLDER)
        )

        return None

    # Only one PDF
    if len(pdf_files) == 1:

        pdf_path = os.path.join(
            PDF_FOLDER,
            pdf_files[0]
        )

        print("\n📄 PDF Found:")
        print(pdf_files[0])

        return pdf_path

    # Multiple PDFs
    print("\n📚 Multiple PDFs found:\n")

    for i, file in enumerate(pdf_files, start=1):

        print(f"{i}. {file}")

    while True:

        try:

            choice = int(
                input("\nEnter PDF number: ")
            )

            if 1 <= choice <= len(pdf_files):

                selected_file = pdf_files[
                    choice - 1
                ]

                return os.path.join(
                    PDF_FOLDER,
                    selected_file
                )

            print("❌ Invalid number.")

        except ValueError:

            print(
                "❌ Please enter a valid number."
            )


# ============================================================
# 3. EXTRACT TEXT FROM PDF
# ============================================================

def extract_pdf_pages(pdf_path):

    print("\n📖 Reading PDF...")

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

        text = page.get_text("text")

        text = text.strip()

        if text:

            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    document.close()

    if len(pages) == 0:

        print(
            "\n❌ No readable text found in PDF."
        )

        print(
            "This may be a scanned/image PDF."
        )

        return None

    print(
        f"✅ Text extracted from {len(pages)} pages."
    )

    return pages


# ============================================================
# 4. CREATE CHUNKS WITH PAGE INFORMATION
# ============================================================

def create_chunks(pages):

    print("\n✂️ Creating text chunks...")

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=900,

        chunk_overlap=150,

        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            ""
        ]
    )

    all_chunks = []

    for page_data in pages:

        page_number = page_data["page"]

        page_text = page_data["text"]

        chunks = splitter.split_text(
            page_text
        )

        for chunk in chunks:

            chunk = chunk.strip()

            if not chunk:
                continue

            all_chunks.append(
                {
                    "text": chunk,
                    "page": page_number
                }
            )

    print(
        f"✅ Total chunks created: {len(all_chunks)}"
    )

    return all_chunks


# ============================================================
# 5. CREATE LANGCHAIN DOCUMENTS
# ============================================================

def create_documents(all_chunks):

    documents = []

    for item in all_chunks:

        from langchain_core.documents import Document

        document = Document(

            page_content=item["text"],

            metadata={
                "page": item["page"]
            }
        )

        documents.append(document)

    return documents


# ============================================================
# 6. LOAD EMBEDDING MODEL
# ============================================================

def load_embeddings():

    print(
        "\n🧠 Loading embedding model..."
    )

    embeddings = HuggingFaceEmbeddings(

        model_name=EMBEDDING_MODEL,

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    print(
        "✅ Embedding model loaded."
    )

    return embeddings


# ============================================================
# 7. CREATE FAISS DATABASE
# ============================================================

def create_vector_database(
    documents,
    embeddings
):

    print(
        "\n🔎 Creating FAISS vector database..."
    )

    # Delete old database
    if os.path.exists(
        VECTORSTORE_FOLDER
    ):

        shutil.rmtree(
            VECTORSTORE_FOLDER
        )

    vectorstore = FAISS.from_documents(

        documents,

        embedding=embeddings
    )

    vectorstore.save_local(
        VECTORSTORE_FOLDER
    )

    print(
        "✅ Vector database created successfully."
    )

    return vectorstore


# ============================================================
# 8. LOAD FAISS DATABASE
# ============================================================

def load_vector_database(
    embeddings
):

    print(
        "\n📂 Loading vector database..."
    )

    vectorstore = FAISS.load_local(

        VECTORSTORE_FOLDER,

        embeddings,

        allow_dangerous_deserialization=True
    )

    print(
        "✅ Vector database loaded."
    )

    return vectorstore


# ============================================================
# 9. LOAD LLAMA
# ============================================================

def load_llm():

    print(
        "\n🤖 Connecting to Llama 3.2..."
    )

    llm = OllamaLLM(

        model=LLM_MODEL,

        temperature=0.1
    )

    print(
        "✅ Llama 3.2 connected."
    )

    return llm


# ============================================================
# 10. CLASSIFY USER QUESTION
# ============================================================

def classify_question(question):

    q = question.lower().strip()

    # Greetings / casual conversation
    casual_words = [

        "hello",
        "hi",
        "hey",
        "how are you",
        "kya haal",
        "kaise ho",
        "good morning",
        "good evening",
        "thanks",
        "thank you"
    ]

    for word in casual_words:

        if word in q:

            return "casual"

    # Summary
    summary_words = [

        "summary",
        "summarize",
        "summarise",
        "overview",
        "briefly explain",
        "short summary"
    ]

    for word in summary_words:

        if word in q:

            return "summary"

    # Topic
    topic_words = [

        "what is the topic",
        "topic of this pdf",
        "topic of the pdf",
        "what is this pdf about",
        "what is this paper about",
        "what is the paper about",
        "title of this pdf",
        "title of the pdf",
        "name of this paper"
    ]

    for word in topic_words:

        if word in q:

            return "topic"

    # Objective
    objective_words = [

        "objective",
        "main objective",
        "purpose of",
        "aim of",
        "goal of",
        "what problem",
        "problem addressed"
    ]

    for word in objective_words:

        if word in q:

            return "objective"

    # Methodology
    methodology_words = [

        "methodology",
        "method used",
        "methods used",
        "approach used",
        "technique used",
        "how was",
        "how did the authors"
    ]

    for word in methodology_words:

        if word in q:

            return "methodology"

    # Conclusion
    conclusion_words = [

        "conclusion",
        "conclude",
        "final result",
        "findings",
        "results of the study",
        "result of the study"
    ]

    for word in conclusion_words:

        if word in q:

            return "conclusion"

    # Contribution
    contribution_words = [

        "contribution",
        "contributions",
        "novelty",
        "novel",
        "what is new"
    ]

    for word in contribution_words:

        if word in q:

            return "contribution"

    return "normal"


# ============================================================
# 11. GET IMPORTANT BEGINNING CHUNKS
# ============================================================

def get_beginning_context(all_chunks):

    # First few chunks usually contain:
    # title
    # abstract
    # introduction
    # objective

    beginning = all_chunks[:6]

    return beginning


# ============================================================
# 12. RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_documents(
    question,
    vectorstore
):

    print(
        "\n🔍 Searching relevant information..."
    )

    try:

        documents = vectorstore.max_marginal_relevance_search(

            question,

            k=RETRIEVAL_K,

            fetch_k=FETCH_K,

            lambda_mult=0.65
        )

    except Exception:

        # Fallback to normal similarity search
        documents = vectorstore.similarity_search(

            question,

            k=RETRIEVAL_K
        )

    return documents


# ============================================================
# 13. REMOVE DUPLICATE DOCUMENTS
# ============================================================

def remove_duplicate_documents(
    documents
):

    unique_documents = []

    seen = set()

    for doc in documents:

        text = doc.page_content.strip()

        if text not in seen:

            seen.add(text)

            unique_documents.append(
                doc
            )

    return unique_documents


# ============================================================
# 14. BUILD CONTEXT
# ============================================================

def build_context(
    question,
    question_type,
    retrieved_documents,
    all_chunks
):

    context_documents = []

    # --------------------------------------------------------
    # For topic/objective/summary type questions,
    # beginning of PDF is very important.
    # --------------------------------------------------------

    if question_type in [

        "topic",
        "objective",
        "summary",
        "methodology"

    ]:

        beginning_documents = []

        for item in all_chunks[:8]:

            from langchain_core.documents import Document

            beginning_documents.append(

                Document(

                    page_content=item["text"],

                    metadata={
                        "page": item["page"]
                    }
                )
            )

        context_documents.extend(
            beginning_documents
        )

    # Add retrieved documents
    context_documents.extend(
        retrieved_documents
    )

    # Remove duplicates
    context_documents = remove_duplicate_documents(
        context_documents
    )

    # Limit context
    context_documents = context_documents[:14]

    # Build final context
    context_parts = []

    for doc in context_documents:

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        text = doc.page_content.strip()

        context_parts.append(

            f"[Page {page}]\n{text}"

        )

    context = "\n\n".join(
        context_parts
    )

    return context, context_documents


# ============================================================
# 15. CREATE PROMPT
# ============================================================

def create_prompt(
    question,
    question_type,
    context,
    conversation_history
):

    history_text = ""

    if conversation_history:

        history_text = "\n".join(

            conversation_history[-6:]
        )

    prompt = f"""
You are a helpful and accurate PDF research assistant.

The user has uploaded a PDF.

Your task is to answer questions using the
information contained in the PDF.

========================
IMPORTANT INSTRUCTIONS
========================

1. Use the PDF context as your primary source.

2. Understand the meaning of the information.
   Do not depend only on exact keyword matching.

3. If information is available in different parts
   of the PDF, combine those parts to create a
   complete answer.

4. Give natural, clear and human-readable answers.

5. For simple questions, give a simple answer.

6. For technical questions, explain the concept
   clearly and then give relevant details.

7. For "what is the topic" questions, identify the
   paper's main subject using the title, abstract,
   introduction and overall content.

8. For summary questions, provide a concise summary
   covering the main topic, objective, approach and
   important findings when available.

9. For objective questions, explain what the research
   is trying to achieve or solve.

10. For methodology questions, explain the approach,
    framework, methods or techniques used.

11. For contribution questions, identify the important
    contribution or novelty described in the PDF.

12. For conclusion/findings questions, summarize the
    results or conclusions described in the PDF.

13. Do NOT invent information.

14. Do NOT use outside knowledge to create facts about
    the research.

15. If the PDF does not contain enough information to
    answer the question, honestly say:

    "I could not find enough information about this
    in the uploaded PDF."

16. If the question is completely unrelated to the PDF,
    politely tell the user that it is outside the scope
    of the uploaded PDF.

17. If the user asks a follow-up question such as
    "why?", "how?", "explain this", or "what does this
    mean?", use the previous conversation and PDF
    context to understand what they are referring to.

18. Never mention that you are an AI unless necessary.

19. Do not unnecessarily repeat the question.

20. When useful, mention the PDF page number.

========================
QUESTION TYPE
========================

{question_type}

========================
PREVIOUS CONVERSATION
========================

{history_text}

========================
PDF CONTEXT
========================

{context}

========================
USER QUESTION
========================

{question}

========================
ANSWER
========================

Give the best possible answer based on the PDF.
"""


    return prompt


# ============================================================
# 16. GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    vectorstore,
    llm,
    all_chunks,
    conversation_history
):

    question_type = classify_question(
        question
    )

    # --------------------------------------------------------
    # Casual question
    # --------------------------------------------------------

    if question_type == "casual":

        return (
            "I'm ready to help you with questions "
            "about the uploaded PDF. "
            "Ask me anything related to the PDF."
        )

    # --------------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------------

    retrieved_documents = retrieve_documents(

        question,

        vectorstore
    )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context, context_documents = build_context(

        question,

        question_type,

        retrieved_documents,

        all_chunks
    )

    # --------------------------------------------------------
    # Create prompt
    # --------------------------------------------------------

    prompt = create_prompt(

        question,

        question_type,

        context,

        conversation_history
    )

    print(
        "🤖 Generating answer..."
    )

    # --------------------------------------------------------
    # Ask Llama
    # --------------------------------------------------------

    answer = llm.invoke(
        prompt
    )

    answer = str(answer).strip()

    # --------------------------------------------------------
    # Add source pages
    # --------------------------------------------------------

    pages = []

    for doc in context_documents:

        page = doc.metadata.get(
            "page"
        )

        if page is not None:

            pages.append(
                int(page)
            )

    pages = sorted(
        set(pages)
    )

    if pages:

        if len(pages) <= 5:

            source_text = ", ".join(
                str(page)
                for page in pages
            )

            answer += (
                f"\n\n📄 Source page(s): "
                f"{source_text}"
            )

    return answer


# ============================================================
# 17. START CHATBOT
# ============================================================

def start_chatbot(
    vectorstore,
    llm,
    all_chunks
):

    print("\n")
    print("=" * 65)
    print("📚 PDF RAG CHATBOT")
    print("=" * 65)

    print(
        "\nAsk questions about your PDF."
    )

    print(
        "Type 'exit' to close."
    )

    print(
        "Type 'clear' to clear conversation history."
    )

    conversation_history = []

    while True:

        print()

        question = input(
            "You: "
        ).strip()

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if question.lower() == "exit":

            print(
                "\n👋 Chatbot closed."
            )

            break

        # ----------------------------------------------------
        # Clear history
        # ----------------------------------------------------

        if question.lower() == "clear":

            conversation_history = []

            print(
                "\n🧹 Conversation history cleared."
            )

            continue

        # ----------------------------------------------------
        # Empty question
        # ----------------------------------------------------

        if not question:

            print(
                "Please enter a question."
            )

            continue

        # ----------------------------------------------------
        # Generate answer
        # ----------------------------------------------------

        try:

            answer = generate_answer(

                question,

                vectorstore,

                llm,

                all_chunks,

                conversation_history
            )

            print("\nBot:")
            print(answer)

            # Save conversation
            conversation_history.append(
                f"User: {question}"
            )

            conversation_history.append(
                f"Assistant: {answer}"
            )

        except Exception as error:

            print(
                "\n❌ Error while generating answer:"
            )

            print(error)


# ============================================================
# 18. MAIN FUNCTION
# ============================================================

def main():

    print("\n")
    print("=" * 65)
    print("🚀 PDF RAG SYSTEM")
    print("=" * 65)

    # --------------------------------------------------------
    # Create PDF folder
    # --------------------------------------------------------

    if not os.path.exists(
        PDF_FOLDER
    ):

        os.makedirs(
            PDF_FOLDER
        )

    # --------------------------------------------------------
    # Find PDF
    # --------------------------------------------------------

    pdf_path = find_pdf()

    if pdf_path is None:

        return

    # --------------------------------------------------------
    # Extract PDF
    # --------------------------------------------------------

    pages = extract_pdf_pages(
        pdf_path
    )

    if pages is None:

        return

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    all_chunks = create_chunks(
        pages
    )

    if not all_chunks:

        print(
            "\n❌ No chunks created."
        )

        return

    # --------------------------------------------------------
    # Create LangChain Documents
    # --------------------------------------------------------

    documents = create_documents(
        all_chunks
    )

    # --------------------------------------------------------
    # Load Embeddings
    # --------------------------------------------------------

    embeddings = load_embeddings()

    # --------------------------------------------------------
    # Create FAISS
    # --------------------------------------------------------

    vectorstore = create_vector_database(

        documents,

        embeddings
    )

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    vectorstore = load_vector_database(

        embeddings
    )

    # --------------------------------------------------------
    # Load Llama
    # --------------------------------------------------------

    llm = load_llm()

    # --------------------------------------------------------
    # Start chatbot
    # --------------------------------------------------------

    start_chatbot(

        vectorstore,

        llm,

        all_chunks
    )


# ============================================================
# 19. PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()