#!/usr/bin/env python3
"""
SOP QA Tool - Interactive Demo Script

This script demonstrates the key capabilities of the SOP QA Tool including:
- Document ingestion and processing
- Vector search and retrieval
- Question answering with confidence scoring
- System health monitoring

Run this after setting up the system to showcase functionality.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

console = Console()

API_BASE_URL = "http://localhost:8000"
UI_URL = "http://localhost:8501"

class SOPQADemo:
    """Interactive demo for SOP QA Tool"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 30
    
    def check_system_health(self) -> bool:
        """Check if the system is running and healthy"""
        try:
            response = self.session.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                health_data = response.json()
                return health_data.get("status") == "healthy"
        except Exception:
            pass
        return False
    
    def display_system_status(self):
        """Display current system status"""
        console.print("\n🔍 [bold blue]System Health Check[/bold blue]")
        
        try:
            response = self.session.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                health_data = response.json()
                
                table = Table(title="System Status")
                table.add_column("Component", style="cyan")
                table.add_column("Status", style="green")
                table.add_column("Details", style="yellow")
                
                table.add_row("Overall", health_data["status"], health_data["mode"])
                
                for component, details in health_data["components"].items():
                    status = details["status"]
                    info = details["details"]
                    table.add_row(component.replace("_", " ").title(), status, info)
                
                console.print(table)
                return True
            else:
                console.print("❌ [red]API not responding[/red]")
                return False
                
        except Exception as e:
            console.print(f"❌ [red]System check failed: {e}[/red]")
            return False
    
    def demonstrate_document_ingestion(self):
        """Demonstrate document ingestion capabilities"""
        console.print("\n📄 [bold blue]Document Ingestion Demo[/bold blue]")
        
        # Check available test documents
        test_docs_dir = Path("test_documents")
        if not test_docs_dir.exists():
            console.print("❌ [red]Test documents directory not found[/red]")
            return
        
        test_files = list(test_docs_dir.glob("*.md"))
        if not test_files:
            console.print("❌ [red]No test documents found[/red]")
            return
        
        console.print(f"📁 Found {len(test_files)} test documents:")
        for i, file in enumerate(test_files, 1):
            console.print(f"  {i}. {file.name}")
        
        # Simulate file upload (in real demo, this would upload via API)
        console.print("\n🚀 [green]Documents are already ingested and ready for querying![/green]")
        
        # Show vector store stats
        try:
            response = self.session.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                health_data = response.json()
                vector_store_info = health_data["components"].get("vector_store", {})
                details = vector_store_info.get("details", "")
                console.print(f"📊 Vector Store: {details}")
        except Exception:
            pass
    
    def demonstrate_question_answering(self):
        """Demonstrate Q&A capabilities with sample questions"""
        console.print("\n💬 [bold blue]Question Answering Demo[/bold blue]")
        
        sample_questions = [
            "What PPE is required in the welding area?",
            "What are the fire emergency procedures?",
            "How do I perform lockout tagout?",
            "What safety training is required for new employees?",
            "What equipment is needed for quality control?"
        ]
        
        console.print("🎯 [yellow]Sample Questions Available:[/yellow]")
        for i, question in enumerate(sample_questions, 1):
            console.print(f"  {i}. {question}")
        
        while True:
            console.print("\n" + "="*60)
            choice = Prompt.ask(
                "Choose a question number (1-5) or type 'custom' for your own question, 'quit' to exit",
                choices=[str(i) for i in range(1, 6)] + ["custom", "quit"]
            )
            
            if choice == "quit":
                break
            elif choice == "custom":
                question = Prompt.ask("Enter your question")
            else:
                question = sample_questions[int(choice) - 1]
            
            self.ask_question(question)
    
    def ask_question(self, question: str):
        """Ask a question and display the response"""
        console.print(f"\n❓ [bold cyan]Question:[/bold cyan] {question}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing question...", total=None)
            
            try:
                payload = {
                    "question": question,
                    "top_k": 3
                }
                
                start_time = time.time()
                response = self.session.post(f"{API_BASE_URL}/ask", json=payload)
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    progress.remove_task(task)
                    
                    # Display answer
                    answer_panel = Panel(
                        result["answer"],
                        title="🤖 Answer",
                        border_style="green"
                    )
                    console.print(answer_panel)
                    
                    # Display confidence
                    confidence = result["confidence"]
                    confidence_level = result["confidence_level"]
                    
                    if confidence_level == "high":
                        confidence_color = "green"
                        confidence_icon = "🟢"
                    elif confidence_level == "medium":
                        confidence_color = "yellow"
                        confidence_icon = "🟡"
                    else:
                        confidence_color = "red"
                        confidence_icon = "🔴"
                    
                    console.print(f"\n{confidence_icon} [bold {confidence_color}]Confidence: {confidence:.2f} ({confidence_level})[/bold {confidence_color}]")
                    console.print(f"⏱️  Processing time: {processing_time:.2f}s")
                    
                    # Display citations
                    if result.get("citations"):
                        console.print("\n📚 [bold blue]Sources:[/bold blue]")
                        for i, citation in enumerate(result["citations"], 1):
                            console.print(f"  {i}. Document: {citation['doc_id']}")
                            console.print(f"     Text: {citation['text_snippet'][:100]}...")
                    
                else:
                    progress.remove_task(task)
                    console.print(f"❌ [red]Error: {response.status_code} - {response.text}[/red]")
                    
            except Exception as e:
                progress.remove_task(task)
                console.print(f"❌ [red]Request failed: {e}[/red]")
    
    def show_system_capabilities(self):
        """Display system capabilities and architecture"""
        console.print("\n🏗️ [bold blue]System Architecture & Capabilities[/bold blue]")
        
        capabilities = [
            "🤖 **AI-Powered Q&A**: Advanced language models with RAG architecture",
            "📄 **Multi-Format Support**: PDF, DOCX, HTML, TXT, Markdown files",
            "🔍 **Intelligent Search**: Vector similarity search with metadata filtering",
            "🎯 **Manufacturing Focus**: Specialized for SOPs and safety procedures",
            "🔄 **Dual Mode**: AWS cloud services OR local offline operation",
            "🔒 **Enterprise Security**: Input validation, PII redaction, SSRF protection",
            "📊 **Real-time Monitoring**: Health checks and performance metrics",
            "🚀 **Production Ready**: Comprehensive testing and deployment automation"
        ]
        
        for capability in capabilities:
            console.print(f"  {capability}")
        
        console.print(f"\n🌐 **Access Points:**")
        console.print(f"  • Web UI: {UI_URL}")
        console.print(f"  • API Docs: {API_BASE_URL}/docs")
        console.print(f"  • Health Check: {API_BASE_URL}/health")
    
    def run_demo(self):
        """Run the complete demo"""
        console.print(Panel.fit(
            "🏭 [bold blue]SOP QA Tool - Interactive Demo[/bold blue]\n"
            "Intelligent Manufacturing Document Assistant",
            border_style="blue"
        ))
        
        # Check system health
        if not self.check_system_health():
            console.print("\n❌ [red]System is not running or unhealthy![/red]")
            console.print("Please ensure both API and UI are running:")
            console.print("  • API: python -m uvicorn sop_qa_tool.api.main:app --reload")
            console.print("  • UI: streamlit run sop_qa_tool/ui/streamlit_app.py")
            return
        
        # Display system status
        self.display_system_status()
        
        # Show capabilities
        self.show_system_capabilities()
        
        # Document ingestion demo
        self.demonstrate_document_ingestion()
        
        # Interactive Q&A
        if Confirm.ask("\nWould you like to try the question answering system?"):
            self.demonstrate_question_answering()
        
        console.print("\n🎉 [bold green]Demo completed![/bold green]")
        console.print("Visit the web interface for full functionality:")
        console.print(f"🌐 {UI_URL}")

def main():
    """Main demo function"""
    try:
        # Check if rich is available
        demo = SOPQADemo()
        demo.run_demo()
    except ImportError:
        print("📦 Installing required demo dependencies...")
        import subprocess
        subprocess.check_call(["pip", "install", "rich"])
        
        # Retry after installation
        demo = SOPQADemo()
        demo.run_demo()
    except KeyboardInterrupt:
        console.print("\n👋 Demo interrupted by user")
    except Exception as e:
        console.print(f"\n❌ Demo failed: {e}")

if __name__ == "__main__":
    main()