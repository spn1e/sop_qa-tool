"""
Streamlit Frontend Implementation

Main Streamlit application providing a chat interface for the SOP Q&A Tool
with document upload, filtering, and export capabilities.

Requirements: 5.1, 5.2, 5.3, 5.4, 4.2, 4.3, 7.3
"""

import asyncio
import io
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import requests
import streamlit as st
from pathlib import Path

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sop_qa_tool.config.settings import get_settings, OperationMode


# Configuration
API_BASE_URL = "http://localhost:8000"
SETTINGS = get_settings()


class APIClient:
    """Client for communicating with the FastAPI backend"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 30
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def ask_question(self, question: str, filters: Optional[Dict] = None, top_k: int = 5) -> Dict[str, Any]:
        """Ask a question to the API"""
        payload = {
            "question": question,
            "filters": filters or {},
            "top_k": top_k
        }
        response = self.session.post(f"{self.base_url}/ask", json=payload)
        response.raise_for_status()
        return response.json()
    
    def ingest_urls(self, urls: List[str], use_ocr: bool = True, extract_ontology: bool = True) -> Dict[str, Any]:
        """Ingest documents from URLs"""
        payload = {
            "urls": urls,
            "use_ocr": use_ocr,
            "extract_ontology": extract_ontology
        }
        response = self.session.post(f"{self.base_url}/ingest/urls", json=payload)
        response.raise_for_status()
        return response.json()
    
    def ingest_files(self, files: List[tuple]) -> Dict[str, Any]:
        """Ingest uploaded files"""
        files_data = []
        for filename, file_content in files:
            files_data.append(("files", (filename, file_content, "application/octet-stream")))
        
        response = self.session.post(f"{self.base_url}/ingest/files", files=files_data)
        response.raise_for_status()
        return response.json()
    
    def get_ingestion_status(self, task_id: str) -> Dict[str, Any]:
        """Get ingestion task status"""
        response = self.session.get(f"{self.base_url}/ingest/{task_id}/status")
        response.raise_for_status()
        return response.json()
    
    def list_sources(self) -> Dict[str, Any]:
        """List all document sources"""
        response = self.session.get(f"{self.base_url}/sources")
        response.raise_for_status()
        return response.json()
    
    def delete_source(self, doc_id: str) -> Dict[str, Any]:
        """Delete a document source"""
        response = self.session.delete(f"{self.base_url}/sources/{doc_id}")
        response.raise_for_status()
        return response.json()
    
    def reindex_documents(self) -> Dict[str, Any]:
        """Trigger document reindexing"""
        response = self.session.post(f"{self.base_url}/reindex")
        response.raise_for_status()
        return response.json()


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient()
    
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = SETTINGS.mode.value
    
    if "ingestion_tasks" not in st.session_state:
        st.session_state.ingestion_tasks = {}
    
    if "filters" not in st.session_state:
        st.session_state.filters = {}
    
    if "export_data" not in st.session_state:
        st.session_state.export_data = []
    
    if "success_notifications" not in st.session_state:
        st.session_state.success_notifications = []


def render_success_notifications():
    """Render success notifications at the top of the app"""
    if st.session_state.success_notifications:
        for i, notification in enumerate(st.session_state.success_notifications):
            col1, col2 = st.columns([10, 1])
            with col1:
                st.success(notification["message"])
            with col2:
                if st.button("✕", key=f"dismiss_{i}", help="Dismiss notification"):
                    st.session_state.success_notifications.remove(notification)
                    st.rerun()


def render_header():
    """Render the application header with mode switching"""
    st.set_page_config(
        page_title="SOP Q&A Tool",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Display success notifications at the top
    render_success_notifications()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.title("🏭 SOP Q&A Tool")
        st.caption("Automated Research & Q/A Tool for Factory SOPs")
    
    with col2:
        # Mode switching
        current_mode = st.selectbox(
            "Operation Mode",
            options=["aws", "local"],
            index=0 if st.session_state.current_mode == "aws" else 1,
            help="Switch between AWS cloud services and local operation"
        )
        
        if current_mode != st.session_state.current_mode:
            st.session_state.current_mode = current_mode
            st.rerun()
    
    with col3:
        # Health status
        health = st.session_state.api_client.health_check()
        if health.get("status") == "healthy":
            st.success("🟢 System Healthy")
        else:
            st.error("🔴 System Issues")
            if st.button("Show Details"):
                st.json(health)


def render_sidebar():
    """Render the sidebar with document management and filtering"""
    with st.sidebar:
        st.header("📚 Document Management")
        
        # Document ingestion section
        with st.expander("📥 Add Documents", expanded=False):
            render_document_ingestion()
        
        # Document sources section
        with st.expander("📋 Document Sources", expanded=False):
            render_document_sources()
        
        # Filtering section
        st.header("🔍 Search Filters")
        render_filters()
        
        # System administration
        with st.expander("⚙️ System Admin", expanded=False):
            render_admin_controls()


def render_document_ingestion():
    """Render document ingestion interface"""
    st.subheader("Add New Documents")
    
    # URL ingestion
    st.write("**From URLs:**")
    urls_text = st.text_area(
        "Enter URLs (one per line)",
        placeholder="https://example.com/sop1.pdf\nhttps://example.com/sop2.docx",
        height=100
    )
    
    col1, col2 = st.columns(2)
    with col1:
        use_ocr = st.checkbox("Use OCR", value=True, help="Extract text from scanned documents")
    with col2:
        extract_ontology = st.checkbox("Extract Structure", value=True, help="Extract SOP ontology information")
    
    if st.button("Ingest URLs", disabled=not urls_text.strip()):
        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
        try:
            result = st.session_state.api_client.ingest_urls(urls, use_ocr, extract_ontology)
            task_id = result["task_id"]
            st.session_state.ingestion_tasks[task_id] = {
                "type": "urls",
                "urls": urls,
                "started_at": datetime.now(),
                "status": "running"
            }
            st.success(f"🚀 **Ingestion started!**\n\n📋 Processing {len(urls)} URL(s)\n⏱️ Estimated time: {result.get('estimated_time_minutes', 'unknown')} minutes\n🆔 Task ID: {task_id}")
            st.info("📊 Check the status below for real-time progress updates")
        except Exception as e:
            st.error(f"❌ **Failed to start ingestion:** {str(e)}")
    
    st.divider()
    
    # File upload
    st.write("**From Files:**")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "html", "txt"],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, HTML, or TXT files"
    )
    
    if uploaded_files and st.button("Ingest Files"):
        try:
            files_data = []
            file_names = []
            for uploaded_file in uploaded_files:
                files_data.append((uploaded_file.name, uploaded_file.getvalue()))
                file_names.append(uploaded_file.name)
            
            result = st.session_state.api_client.ingest_files(files_data)
            task_id = result["task_id"]
            st.session_state.ingestion_tasks[task_id] = {
                "type": "files",
                "files": file_names,
                "started_at": datetime.now(),
                "status": "running"
            }
            st.success(f"🚀 **File ingestion started!**\n\n📁 Processing {len(file_names)} file(s): {', '.join(file_names)}\n⏱️ Estimated time: {result.get('estimated_time_minutes', 'unknown')} minutes\n🆔 Task ID: {task_id}")
            st.info("📊 Check the status below for real-time progress updates")
        except Exception as e:
            st.error(f"❌ **Failed to start file ingestion:** {str(e)}")
    
    # Show active ingestion tasks
    if st.session_state.ingestion_tasks:
        st.divider()
        st.write("**Active Tasks:**")
        render_ingestion_status()


def render_ingestion_status():
    """Render ingestion task status"""
    tasks_to_remove = []
    
    for task_id, task_info in st.session_state.ingestion_tasks.items():
        if task_info["status"] == "running":
            try:
                status = st.session_state.api_client.get_ingestion_status(task_id)
                
                # Create progress display
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{task_info['type'].title()}:** {status['message']}")
                    progress = status.get("progress", 0.0)
                    st.progress(progress)
                    
                    # Show processing details
                    if status.get("documents_processed", 0) > 0:
                        st.caption(f"📄 Processed: {status['documents_processed']}/{status.get('documents_total', 0)} documents")
                    
                    if status["errors"]:
                        with st.expander("⚠️ Errors"):
                            for error in status["errors"]:
                                st.error(error)
                
                with col2:
                    if status["status"] in ["completed", "failed"]:
                        if st.button("Clear", key=f"clear_{task_id}"):
                            tasks_to_remove.append(task_id)
                
                # Update task status
                task_info["status"] = status["status"]
                task_info["final_status"] = status  # Store final status for success message
                
                if status["status"] == "completed":
                    # Enhanced success message
                    docs_processed = status.get("documents_processed", 0)
                    if task_info["type"] == "files":
                        file_names = ", ".join(task_info.get("files", []))
                        success_msg = f"🎉 **Document(s) ingested successfully!**\n\n📁 Files: {file_names}\n📊 {docs_processed} document(s) processed and indexed"
                        st.success(success_msg)
                        
                        # Add to global notifications
                        notification_msg = f"🎉 **Files successfully ingested:** {file_names} ({docs_processed} documents processed)"
                        if notification_msg not in [n["message"] for n in st.session_state.success_notifications]:
                            st.session_state.success_notifications.append({
                                "message": notification_msg,
                                "timestamp": datetime.now()
                            })
                    else:
                        urls_count = len(task_info.get("urls", []))
                        success_msg = f"🎉 **Document(s) ingested successfully!**\n\n🔗 {urls_count} URL(s) processed\n📊 {docs_processed} document(s) processed and indexed"
                        st.success(success_msg)
                        
                        # Add to global notifications
                        notification_msg = f"🎉 **URLs successfully ingested:** {urls_count} URL(s) ({docs_processed} documents processed)"
                        if notification_msg not in [n["message"] for n in st.session_state.success_notifications]:
                            st.session_state.success_notifications.append({
                                "message": notification_msg,
                                "timestamp": datetime.now()
                            })
                    
                    # Show additional success details
                    if status.get("result"):
                        result = status["result"]
                        processing_time = result.get("processing_time_seconds", 0)
                        st.info(f"⏱️ Processing completed in {processing_time:.1f} seconds")
                        
                elif status["status"] == "failed":
                    st.error("❌ **Document ingestion failed!**")
                    if status["errors"]:
                        st.error(f"Errors: {'; '.join(status['errors'])}")
                
            except Exception as e:
                st.error(f"Failed to get status for task {task_id}: {str(e)}")
                tasks_to_remove.append(task_id)
    
    # Remove completed/failed tasks
    for task_id in tasks_to_remove:
        del st.session_state.ingestion_tasks[task_id]


def render_document_sources():
    """Render document sources management"""
    st.subheader("Manage Sources")
    
    try:
        sources_data = st.session_state.api_client.list_sources()
        sources = sources_data.get("sources", [])
        
        if sources:
            for source in sources:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{source['title']}**")
                    st.caption(f"ID: {source['doc_id']} | Chunks: {source['chunk_count']}")
                with col2:
                    if st.button("🗑️", key=f"delete_{source['doc_id']}", help="Delete source"):
                        try:
                            st.session_state.api_client.delete_source(source['doc_id'])
                            st.success("Source deleted!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete: {str(e)}")
        else:
            st.info("No documents ingested yet")
            
    except Exception as e:
        st.error(f"Failed to load sources: {str(e)}")


def render_filters():
    """Render search filtering controls"""
    st.subheader("Filter Results")
    
    # Role filter
    role_filter = st.selectbox(
        "Role",
        options=["All", "Operator", "QA Inspector", "Supervisor", "Maintenance", "Safety Officer"],
        index=0,
        help="Filter by role responsibility"
    )
    
    # Equipment filter
    equipment_filter = st.text_input(
        "Equipment",
        placeholder="e.g., Filler-01, Conveyor-A",
        help="Filter by equipment name"
    )
    
    # Document type filter
    doc_type_filter = st.selectbox(
        "Document Type",
        options=["All", "SOP", "Work Instruction", "Safety Procedure", "Maintenance Guide"],
        index=0,
        help="Filter by document type"
    )
    
    # Advanced filters
    with st.expander("Advanced Filters"):
        confidence_threshold = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
            help="Minimum confidence score for results"
        )
        
        max_results = st.slider(
            "Max Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Maximum number of results to return"
        )
    
    # Update session state filters
    filters = {}
    if role_filter != "All":
        filters["role"] = role_filter
    if equipment_filter.strip():
        filters["equipment"] = equipment_filter.strip()
    if doc_type_filter != "All":
        filters["document_type"] = doc_type_filter
    
    st.session_state.filters = filters
    st.session_state.confidence_threshold = confidence_threshold
    st.session_state.max_results = max_results


def render_admin_controls():
    """Render system administration controls"""
    st.subheader("System Controls")
    
    if st.button("🔄 Rebuild Index"):
        try:
            result = st.session_state.api_client.reindex_documents()
            st.success("Index rebuild started!")
            st.json(result)
        except Exception as e:
            st.error(f"Failed to start reindex: {str(e)}")
    
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.export_data = []
        st.success("Chat history cleared!")
        st.rerun()


def render_chat_interface():
    """Render the main chat interface"""
    st.header("💬 Ask Questions")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                render_assistant_message(message)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your SOPs..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.api_client.ask_question(
                        question=prompt,
                        filters=st.session_state.filters,
                        top_k=st.session_state.max_results
                    )
                    
                    # Filter by confidence threshold
                    if response["confidence"] < st.session_state.confidence_threshold:
                        response["low_confidence_warning"] = True
                    
                    # Add to messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "response_data": response,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Add to export data
                    st.session_state.export_data.append({
                        "question": prompt,
                        "answer": response["answer"],
                        "confidence": response["confidence"],
                        "confidence_level": response["confidence_level"],
                        "citations": response["citations"],
                        "timestamp": datetime.now().isoformat(),
                        "filters_used": st.session_state.filters.copy()
                    })
                    
                    # Render the response
                    render_assistant_message(st.session_state.messages[-1])
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "error": True,
                        "timestamp": datetime.now().isoformat()
                    })


def render_assistant_message(message: Dict[str, Any]):
    """Render an assistant message with rich formatting"""
    # Main answer
    st.write(message["content"])
    
    if "response_data" in message:
        response_data = message["response_data"]
        
        # Confidence indicator
        confidence = response_data["confidence"]
        confidence_level = response_data["confidence_level"]
        
        if confidence_level == "high":
            confidence_color = "green"
            confidence_icon = "🟢"
        elif confidence_level == "medium":
            confidence_color = "orange"
            confidence_icon = "🟡"
        else:
            confidence_color = "red"
            confidence_icon = "🔴"
        
        st.markdown(f"{confidence_icon} **Confidence:** {confidence:.2f} ({confidence_level})")
        
        # Low confidence warning
        if response_data.get("low_confidence_warning"):
            st.warning("⚠️ Low confidence answer. Please review source documents for verification.")
        
        # Citations
        if response_data.get("citations"):
            with st.expander("📚 Sources & Citations"):
                for i, citation in enumerate(response_data["citations"], 1):
                    st.markdown(f"**Citation {i}:**")
                    st.markdown(f"- Document: `{citation['doc_id']}`")
                    st.markdown(f"- Chunk: `{citation['chunk_id']}`")
                    with st.container():
                        st.markdown("**Relevant text:**")
                        st.info(citation["text_snippet"])
                    st.divider()
        
        # Processing time
        processing_time = response_data.get("processing_time_ms", 0)
        st.caption(f"⏱️ Processed in {processing_time}ms")


def render_export_section():
    """Render export functionality"""
    if not st.session_state.export_data:
        st.info("No Q&A data to export yet. Start asking questions!")
        return
    
    st.header("📤 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV Export
        if st.button("📊 Export as CSV"):
            df = pd.DataFrame(st.session_state.export_data)
            
            # Flatten citations for CSV
            df_export = df.copy()
            df_export["citations_text"] = df_export["citations"].apply(
                lambda x: "; ".join([f"{c['doc_id']}: {c['text_snippet'][:100]}..." for c in x]) if x else ""
            )
            df_export = df_export.drop(columns=["citations"])
            
            csv_buffer = io.StringIO()
            df_export.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"sop_qa_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        # Markdown Export
        if st.button("📝 Export as Markdown"):
            markdown_content = generate_markdown_export()
            
            st.download_button(
                label="Download Markdown",
                data=markdown_content,
                file_name=f"sop_qa_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
    
    # Preview export data
    with st.expander("📋 Preview Export Data"):
        st.dataframe(
            pd.DataFrame(st.session_state.export_data)[
                ["question", "answer", "confidence", "confidence_level", "timestamp"]
            ]
        )


def generate_markdown_export() -> str:
    """Generate markdown export of Q&A data"""
    markdown_lines = [
        "# SOP Q&A Export",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Q&A pairs: {len(st.session_state.export_data)}",
        ""
    ]
    
    for i, item in enumerate(st.session_state.export_data, 1):
        markdown_lines.extend([
            f"## Q&A {i}",
            f"**Timestamp:** {item['timestamp']}",
            f"**Confidence:** {item['confidence']:.2f} ({item['confidence_level']})",
            "",
            f"**Question:** {item['question']}",
            "",
            f"**Answer:** {item['answer']}",
            ""
        ])
        
        if item.get("citations"):
            markdown_lines.append("**Citations:**")
            for j, citation in enumerate(item["citations"], 1):
                markdown_lines.extend([
                    f"{j}. Document: `{citation['doc_id']}` (Chunk: `{citation['chunk_id']}`)",
                    f"   > {citation['text_snippet']}",
                    ""
                ])
        
        if item.get("filters_used"):
            markdown_lines.extend([
                "**Filters Applied:**",
                f"```json",
                json.dumps(item["filters_used"], indent=2),
                "```",
                ""
            ])
        
        markdown_lines.append("---")
        markdown_lines.append("")
    
    return "\n".join(markdown_lines)


def main():
    """Main application entry point"""
    initialize_session_state()
    render_header()
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_chat_interface()
    
    with col2:
        render_export_section()
    
    # Sidebar
    render_sidebar()


if __name__ == "__main__":
    main()