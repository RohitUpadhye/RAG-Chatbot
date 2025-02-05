import os
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split extracted text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save the FAISS vector store locally
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("C:/faiss_index")  

# Function to load the FAISS vector store from local drive
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("C:/faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vector_store

# Function to get conversational chain using Gemini AI
def get_conversational_chain():
    prompt_template = """
    Answer the question as comprehensively and in detail as possible using the provided context. 
    Provide all relevant information and explanations. If the answer is not in the provided context, 
    say "The answer is not available in the provided context." Do not guess or provide incorrect information.
    
    Context: \n{context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2, max_output_tokens=500)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to fetch Google content and links
def fetch_google_content_and_links(query):
    api_key = os.getenv("GOOGLE_API_KEY")
    custom_search_engine_id = os.getenv("GOOGLE_CSE_ID")

    search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={custom_search_engine_id}&key={api_key}"

    response = requests.get(search_url)
    if response.status_code == 200:
        search_results = response.json().get('items', [])
        google_content = []
        google_references = []

        for result in search_results[:4]:
            title = result.get('title')
            link = result.get('link')
            snippet = result.get('snippet')
            
            google_content.append(f"{snippet}")
            google_references.append(f"{title}: {link}")

        return google_content, google_references
    else:
        return ["Failed to fetch Google content"], ["Failed to fetch Google links"]
    
# Function to fetch YouTube videos based on query
def fetch_youtube_videos(query):
    api_key = os.getenv("YOUTUBE_API_KEY")
    search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=3&q={query}&key={api_key}"
    
    response = requests.get(search_url)
    if response.status_code == 200:
        videos = response.json().get('items', [])
        video_links = []
        for video in videos:
            video_id = video.get('id', {}).get('videoId')
            video_title = video.get('snippet', {}).get('title')
            if video_id and video_title:
                video_links.append({
                    'title': video_title,
                    'url': f"https://www.youtube.com/watch?v={video_id}"
                })
        return video_links
    else:
        return [{"title": "Error fetching videos", "url": ""}]

# Function to handle user input and generate response
def user_input(user_question):
    try:
        vector_store = load_vector_store()
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return

    # Perform similarity search
    try:
        docs = vector_store.similarity_search(user_question, k=5)
        if not docs:
            st.warning("No relevant documents found in the vector store.")
            return
    except Exception as e:
        st.error(f"Error during similarity search: {e}")
        return

    # Generate an answer using Gemini AI
    try:
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        answer = response["output_text"]

        # Fetch external resources (YouTube and Google)
        youtube_videos = fetch_youtube_videos(user_question)
        google_content, google_references = fetch_google_content_and_links(user_question)

        return answer, youtube_videos, google_content, google_references
    except Exception as e:
        st.error(f"Error generating the answer: {e}")
        return

# Streamlit Interface
def main():
    st.title("PDF-based Q&A System")

    # Upload PDF files
    pdf_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

    if pdf_files:
        # Process PDFs and create the vector store
        try:
            raw_text = ""
            for pdf in pdf_files:
                raw_text += get_pdf_text([pdf])
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("PDF processing complete.")
        except Exception as e:
            st.error(f"Failed to process PDFs: {e}")

  
     # User Question Input
    user_question = st.text_input("Ask a question based on the uploaded PDFs:")
    
        

    if user_question:
        answer, youtube_videos, google_content, google_references = user_input(user_question)
        
        if answer:
            st.subheader("AI Answer")
            st.write(answer)

            if google_content:
                st.subheader("Google Search Content:")
                for content in google_content:
                    st.write(content)

            if google_references:
                st.subheader("Google References:")
                for reference in google_references:
                    st.write(reference)

            if youtube_videos:
                st.subheader("YouTube Videos:")
                for video in youtube_videos:
                    st.write(f"[{video['title']}]({video['url']})")

if __name__ == "__main__":
    main()
