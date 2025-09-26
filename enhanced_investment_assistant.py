# enhanced_investment_assistant.py
import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import os
import re
import hashlib
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DocumentChunk:
    content: str
    source: str
    page: int
    chunk_id: str
    embedding: np.ndarray = None

class EnhancedPDFProcessor:
    """Advanced PDF processing with multiple extraction methods"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def extract_text_comprehensive(self, pdf_file) -> Dict:
        """Extract text using multiple methods for better accuracy"""
        results = {
            'text': '',
            'tables': [],
            'metadata': {},
            'success': False
        }
        
        try:
            # Method 1: pdfplumber (best for tables and layout)
            with pdfplumber.open(pdf_file) as pdf:
                text_parts = []
                tables = []
                
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"\n--- Page {page_num + 1} ---\n{page_text}")
                    
                    # Extract tables
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            tables.append({
                                'page': page_num + 1,
                                'data': df,
                                'raw': table
                            })
                
                results['text'] = '\n'.join(text_parts)
                results['tables'] = tables
                results['success'] = True
                
        except Exception as e:
            st.warning(f"pdfplumber failed: {e}")
            
            # Fallback: PyMuPDF
            try:
                pdf_file.seek(0)
                pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
                text_parts = []
                
                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    text = page.get_text()
                    if text.strip():
                        text_parts.append(f"\n--- Page {page_num + 1} ---\n{text}")
                
                results['text'] = '\n'.join(text_parts)
                results['success'] = True
                pdf_document.close()
                
            except Exception as e2:
                st.error(f"All PDF extraction methods failed: {e2}")
        
        return results

class SmartContextRetriever:
    """Intelligent context retrieval using embeddings and semantic search"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []
        self.index = None
        
    def create_chunks(self, documents: List[Dict]) -> List[DocumentChunk]:
        """Create semantic chunks from documents"""
        chunks = []
        
        for doc in documents:
            text = doc['content']
            source = doc['filename']
            
            # Split into semantic chunks (paragraphs + sliding window)
            paragraphs = self._split_into_paragraphs(text)
            
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) > 50:  # Only meaningful chunks
                    chunk_id = hashlib.md5(f"{source}_{i}_{paragraph[:100]}".encode()).hexdigest()
                    
                    chunk = DocumentChunk(
                        content=paragraph,
                        source=source,
                        page=self._extract_page_number(paragraph),
                        chunk_id=chunk_id
                    )
                    chunks.append(chunk)
        
        self.chunks = chunks
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into meaningful paragraphs"""
        # Split by double newlines, but also consider financial document patterns
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Further split long paragraphs
        refined_paragraphs = []
        for para in paragraphs:
            if len(para) > 1000:
                # Split by sentences for long paragraphs
                sentences = re.split(r'[.!?]+', para)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) < 800:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            refined_paragraphs.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                
                if current_chunk:
                    refined_paragraphs.append(current_chunk.strip())
            else:
                refined_paragraphs.append(para)
        
        return [p for p in refined_paragraphs if len(p.strip()) > 50]
    
    def _extract_page_number(self, text: str) -> int:
        """Extract page number from text"""
        match = re.search(r'--- Page (\d+) ---', text)
        return int(match.group(1)) if match else 1
    
    def build_index(self):
        """Build FAISS index for semantic search"""
        if not self.chunks:
            return
        
        # Generate embeddings
        texts = [chunk.content for chunk in self.chunks]
        embeddings = self.embedding_model.encode(texts)
        
        # Store embeddings in chunks
        for chunk, embedding in zip(self.chunks, embeddings):
            chunk.embedding = embedding
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
    
    def retrieve_relevant_context(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Retrieve most relevant chunks for a query"""
        if not self.index or not self.chunks:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        relevant_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if score > 0.3:  # Similarity threshold
                chunk = self.chunks[idx]
                relevant_chunks.append(chunk)
        
        return relevant_chunks

class InvestmentAnalysisAgent:
    """Specialized agent for investment analysis"""
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.analysis_prompts = {
            'portfolio_overview': """
            Analyze this investment portfolio data and provide:
            1. Overall portfolio composition and diversification
            2. Risk assessment across asset classes
            3. Performance highlights and concerns
            4. Key financial metrics and ratios
            
            Explain in simple terms suitable for both beginners and experienced investors.
            """,
            
            'risk_analysis': """
            Focus on risk analysis for this investment data:
            1. Identify key risk factors and exposures
            2. Assess risk-adjusted returns
            3. Evaluate portfolio volatility and correlation
            4. Suggest risk mitigation strategies
            
            Use clear explanations and practical recommendations.
            """,
            
            'performance_analysis': """
            Analyze the performance data provided:
            1. Historical performance trends and patterns
            2. Benchmark comparisons and relative performance
            3. Key performance drivers and detractors
            4. Future outlook and recommendations
            
            Provide actionable insights for investment decisions.
            """,
            
            'general': """
            You are an expert investment advisor analyzing financial documents. 
            Provide clear, actionable insights based on the context provided.
            Explain complex concepts simply and offer practical recommendations.
            """
        }
    
    def analyze_query(self, query: str) -> str:
        """Determine the type of analysis needed"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['risk', 'volatility', 'exposure', 'correlation']):
            return 'risk_analysis'
        elif any(word in query_lower for word in ['performance', 'return', 'benchmark', 'trend']):
            return 'performance_analysis'
        elif any(word in query_lower for word in ['portfolio', 'allocation', 'diversification', 'overview']):
            return 'portfolio_overview'
        else:
            return 'general'
    
    def generate_response(self, query: str, context_chunks: List[DocumentChunk]) -> str:
        """Generate intelligent response using context and specialized prompts"""
        
        # Determine analysis type
        analysis_type = self.analyze_query(query)
        system_prompt = self.analysis_prompts[analysis_type]
        
        # Prepare context
        context_text = self._prepare_context(context_chunks)
        
        # Enhanced system prompt
        full_system_prompt = f"""
        {system_prompt}
        
        CONTEXT FROM UPLOADED DOCUMENTS:
        {context_text}
        
        GUIDELINES:
        - Base your analysis primarily on the provided document context
        - If information is missing, clearly state what additional data would be helpful
        - Use specific numbers and data points from the documents when available
        - Provide actionable recommendations
        - Explain technical terms in simple language
        - Structure your response clearly with headers if appropriate
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",  # Using GPT-4 for better analysis
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=1500,
                temperature=0.3  # Lower temperature for more focused responses
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"I encountered an error while analyzing your request: {str(e)}"
    
    def _prepare_context(self, chunks: List[DocumentChunk]) -> str:
        """Prepare context from retrieved chunks"""
        if not chunks:
            return "No relevant context found in uploaded documents."
        
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"""
            Source: {chunk.source} (Page {chunk.page})
            Content: {chunk.content}
            ---
            """)
        
        return '\n'.join(context_parts)

def main():
    st.set_page_config(
        page_title="InvestAssist",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("🏦 InvestAssist")
    st.markdown("Upload portfolio documents and get AI powered analysis")
    
    # Initialize session state
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'retriever' not in st.session_state:
        st.session_state.retriever = SmartContextRetriever()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'analysis_agent' not in st.session_state:
        # Try Streamlit secrets first, fallback to environment variable for local development
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except:
            api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        st.session_state.analysis_agent = InvestmentAnalysisAgent(client)
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📄Your documents")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Investment Documents (PDF)",
            type="pdf",
            accept_multiple_files=True,
            help="Upload portfolio statements, research reports, prospectuses, etc."
        )
        
        if uploaded_files:
            processor = EnhancedPDFProcessor()
            
            for uploaded_file in uploaded_files:
                # Check if already processed
                if not any(doc['filename'] == uploaded_file.name for doc in st.session_state.documents):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        result = processor.extract_text_comprehensive(uploaded_file)
                        
                        if result['success']:
                            st.session_state.documents.append({
                                'filename': uploaded_file.name,
                                'content': result['text'],
                                'tables': result['tables'],
                                'processed_at': datetime.now()
                            })
                            st.success(f"✅ {uploaded_file.name}")
                        else:
                            st.error(f"❌ Failed to process {uploaded_file.name}")
            
            # Build search index
            if st.session_state.documents:
                with st.spinner("Building intelligent search index..."):
                    chunks = st.session_state.retriever.create_chunks(st.session_state.documents)
                    st.session_state.retriever.build_index()
                    st.success(f"🔍 Indexed {len(chunks)} content chunks")
        
        # Document status
        st.subheader("📋 Loaded Documents")
        for doc in st.session_state.documents:
            st.write(f"• {doc['filename']}")
            st.caption(f"Processed: {doc['processed_at'].strftime('%H:%M:%S')}")
        
        if st.button("🗑️ Clear All"):
            st.session_state.documents = []
            st.session_state.retriever = SmartContextRetriever()
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        #st.subheader("💬 Investment Analysis Chat")
        # Quick analysis buttons
        st.markdown("**Quick Analysis Options:**")
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("📊 Portfolio Overview"):
                query = "Provide a comprehensive overview of my investment portfolio including allocation, diversification, and key metrics."
        with col_btn2:
            if st.button("⚠️ Risk Analysis"):
                query = "Analyze the risk profile of my investments including volatility, correlation, and risk factors."
        with col_btn3:
            if st.button("📈 Performance Analysis"):
                query = "Review the performance of my investments including returns, benchmarks, and trends."
        #with col_btn4:
        #    if st.button("🎯 Recommendations"):
        #        query = "Based on my portfolio, what are your key recommendations for optimization?"
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your investments..."):
            query = prompt
        
        # Process query if available
        if 'query' in locals() and query:
            st.session_state.messages.append({"role": "user", "content": query})
            
            with st.chat_message("user"):
                st.markdown(query)
            
            with st.chat_message("assistant"):
                if st.session_state.documents:
                    # Retrieve relevant context
                    relevant_chunks = st.session_state.retriever.retrieve_relevant_context(query, top_k=5)
                    
                    # Generate response
                    with st.spinner("Analyzing your investment documents..."):
                        response = st.session_state.analysis_agent.generate_response(query, relevant_chunks)
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Show sources
                    if relevant_chunks:
                        with st.expander("📖 Sources Used"):
                            for chunk in relevant_chunks:
                                st.write(f"**{chunk.source}** (Page {chunk.page})")
                                st.caption(chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content)
                
                else:
                    st.warning("Please upload some investment documents first to get analysis.")
    
    with col2:
        if st.session_state.documents:
            st.subheader("📊 Document Insights")
            
            # Document statistics
            total_content = sum(len(doc['content']) for doc in st.session_state.documents)
            st.metric("Total Content", f"{total_content:,} chars")
            st.metric("Documents", len(st.session_state.documents))
            st.metric("Search Chunks", len(st.session_state.retriever.chunks))
            
            # Tables found
            total_tables = sum(len(doc.get('tables', [])) for doc in st.session_state.documents)
            if total_tables > 0:
                st.metric("Tables Extracted", total_tables)
                
                if st.button("📋 View Tables"):
                    for doc in st.session_state.documents:
                        if doc.get('tables'):
                            st.subheader(f"Tables from {doc['filename']}")
                            for i, table in enumerate(doc['tables']):
                                st.write(f"Page {table['page']}, Table {i+1}")
                                st.dataframe(table['data'])

if __name__ == "__main__":
    main()
