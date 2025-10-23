"""
Ontology Extraction Service for SOP documents.

This service extracts structured information from SOP documents using LLM-powered
analysis with both AWS Bedrock and local model fallbacks. It handles partial
extractions, validation, and merging of results from multiple chunks.

Requirements: 2.1, 2.2, 2.3, 7.2
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    ClientError = Exception
    BotoCoreError = Exception

from pydantic import ValidationError

from ..models.sop_models import (
    SOPDocument, DocumentChunk, ExtractionResult, ValidationResult,
    ProcedureStep, Risk, Control, RoleResponsibility, Definition,
    ChangeLogEntry, SourceInfo
)
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class OntologyExtractor:
    """
    Extracts structured SOP information from text using LLM analysis.
    
    Supports both AWS Bedrock (primary) and local model (fallback) modes
    for robust operation across different deployment scenarios.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._bedrock_client = None
        self._local_model = None
        self._extraction_prompts = self._load_extraction_prompts()
    
    def _get_bedrock_client(self):
        """Get or create Bedrock client for AWS mode."""
        if self._bedrock_client is None and self.settings.is_aws_mode():
            if not BOTO3_AVAILABLE:
                raise ImportError("boto3 is required for AWS mode but not installed")
            try:
                session = boto3.Session(
                    profile_name=self.settings.aws_profile,
                    region_name=self.settings.aws_region
                )
                self._bedrock_client = session.client('bedrock-runtime')
                logger.info("Initialized Bedrock client")
            except Exception as e:
                logger.error(f"Failed to initialize Bedrock client: {e}")
                raise
        return self._bedrock_client
    
    def _get_local_model(self):
        """Get or create local model for fallback mode."""
        if self._local_model is None and self.settings.is_local_mode():
            try:
                # Import here to avoid dependency issues in AWS mode
                from transformers import pipeline
                
                self._local_model = pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-medium",  # Lightweight model for local use
                    device=-1  # CPU only
                )
                logger.info("Initialized local model")
            except ImportError:
                logger.warning("transformers not available, local extraction disabled")
            except Exception as e:
                logger.error(f"Failed to initialize local model: {e}")
        return self._local_model
    
    def _load_extraction_prompts(self) -> Dict[str, str]:
        """Load extraction prompts for different SOP components."""
        return {
            "system_prompt": """You are an expert at extracting structured information from Standard Operating Procedures (SOPs) and Work Instructions used in manufacturing environments.

Your task is to analyze the provided text and extract structured information according to the specified JSON schema. Focus on manufacturing-specific elements like procedure steps, risks, controls, roles, equipment, and compliance requirements.

Key guidelines:
1. Extract only information that is explicitly stated in the text
2. Use null or empty arrays for missing information - do not invent data
3. Maintain the hierarchical structure of procedure steps (e.g., 1, 1.1, 1.2)
4. Identify roles, equipment, and materials mentioned in each step
5. Extract safety risks and quality controls with their relationships to steps
6. Preserve document metadata like revision numbers and effective dates
7. Return valid JSON that matches the provided schema exactly

If the text doesn't contain SOP-like content, return a minimal structure with just the basic fields populated.""",
            
            "extraction_prompt": """Extract structured SOP information from the following text and return it as JSON matching this schema:

{schema}

Text to analyze:
{text}

Return only valid JSON without any additional text or formatting.""",
            
            "merge_prompt": """You have multiple partial extractions from the same SOP document. Merge them into a single comprehensive structure.

Guidelines for merging:
1. Combine all procedure steps, maintaining proper ordering
2. Merge all risks and controls, removing duplicates
3. Consolidate roles and responsibilities
4. Keep the most complete version of metadata fields
5. Preserve all unique definitions and glossary terms
6. Maintain referential integrity between steps, risks, and controls

Partial extractions to merge:
{extractions}

Return the merged result as valid JSON."""
        }
    
    def extract_from_text(self, text: str, doc_id: str, source_info: SourceInfo) -> ExtractionResult:
        """
        Extract structured SOP information from raw text.
        
        Args:
            text: Raw text content to analyze
            doc_id: Unique document identifier
            source_info: Information about the source document
            
        Returns:
            ExtractionResult with extracted SOP document and metadata
        """
        start_time = time.time()
        
        try:
            # Try AWS Bedrock first if available
            if self.settings.is_aws_mode():
                result = self._extract_with_bedrock(text, doc_id, source_info)
            else:
                result = self._extract_with_local_model(text, doc_id, source_info)
            
            # Validate the extraction
            validation_result = self._validate_extraction(result.sop_document)
            if not validation_result.is_valid:
                result.warnings.extend(validation_result.errors)
            
            result.processing_time_seconds = time.time() - start_time
            logger.info(f"Extraction completed for {doc_id} in {result.processing_time_seconds:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Extraction failed for {doc_id}: {e}")
            return ExtractionResult(
                success=False,
                errors=[f"Extraction failed: {str(e)}"],
                processing_time_seconds=time.time() - start_time
            )
    
    def extract_from_chunks(self, chunks: List[DocumentChunk], doc_id: str, source_info: SourceInfo) -> ExtractionResult:
        """
        Extract structured SOP information from multiple text chunks and merge results.
        
        Args:
            chunks: List of document chunks to process
            doc_id: Unique document identifier
            source_info: Information about the source document
            
        Returns:
            ExtractionResult with merged SOP document
        """
        start_time = time.time()
        partial_extractions = []
        errors = []
        warnings = []
        
        try:
            # Extract from each chunk
            for chunk in chunks:
                try:
                    chunk_result = self.extract_from_text(
                        chunk.chunk_text, 
                        f"{doc_id}_chunk_{chunk.chunk_index}",
                        source_info
                    )
                    
                    if chunk_result.success and chunk_result.sop_document:
                        partial_extractions.append(chunk_result.sop_document)
                    else:
                        warnings.extend(chunk_result.errors)
                        
                except Exception as e:
                    logger.warning(f"Failed to extract from chunk {chunk.chunk_index}: {e}")
                    errors.append(f"Chunk {chunk.chunk_index} extraction failed: {str(e)}")
            
            if not partial_extractions:
                return ExtractionResult(
                    success=False,
                    errors=["No successful extractions from any chunks"] + errors,
                    warnings=warnings,
                    processing_time_seconds=time.time() - start_time
                )
            
            # Merge partial extractions
            merged_sop = self._merge_extractions(partial_extractions, doc_id, source_info)
            
            # Validate merged result
            validation_result = self._validate_extraction(merged_sop)
            if not validation_result.is_valid:
                warnings.extend(validation_result.errors)
            
            return ExtractionResult(
                success=True,
                sop_document=merged_sop,
                chunks=chunks,
                errors=errors,
                warnings=warnings,
                processing_time_seconds=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Chunk-based extraction failed for {doc_id}: {e}")
            return ExtractionResult(
                success=False,
                errors=[f"Chunk extraction failed: {str(e)}"] + errors,
                warnings=warnings,
                processing_time_seconds=time.time() - start_time
            )
    
    def _extract_with_bedrock(self, text: str, doc_id: str, source_info: SourceInfo) -> ExtractionResult:
        """Extract using AWS Bedrock Claude model."""
        try:
            client = self._get_bedrock_client()
            
            # Prepare the prompt with schema
            schema = self._get_sop_schema()
            prompt = self._extraction_prompts["extraction_prompt"].format(
                schema=json.dumps(schema, indent=2),
                text=text[:8000]  # Limit text length for API
            )
            
            # Call Bedrock
            response = client.invoke_model(
                modelId=self.settings.bedrock_model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4000,
                    "system": self._extraction_prompts["system_prompt"],
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            extracted_text = response_body['content'][0]['text']
            
            # Parse JSON from response
            try:
                extracted_data = json.loads(extracted_text)
            except json.JSONDecodeError:
                # Try to extract JSON from response if wrapped in markdown
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', extracted_text, re.DOTALL)
                if json_match:
                    extracted_data = json.loads(json_match.group(1))
                else:
                    raise ValueError("Could not parse JSON from model response")
            
            # Create SOP document
            sop_doc = self._create_sop_document(extracted_data, doc_id, source_info)
            
            return ExtractionResult(
                success=True,
                sop_document=sop_doc,
                warnings=[]
            )
            
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Bedrock API error: {e}")
            # Fallback to local model if available
            if self.settings.is_local_mode():
                logger.info("Falling back to local model")
                return self._extract_with_local_model(text, doc_id, source_info)
            else:
                raise
        except Exception as e:
            logger.error(f"Bedrock extraction error: {e}")
            raise
    
    def _extract_with_local_model(self, text: str, doc_id: str, source_info: SourceInfo) -> ExtractionResult:
        """Extract using local model as fallback."""
        try:
            # For local mode, we'll use a simpler rule-based extraction
            # as a fallback when full LLM models aren't available
            extracted_data = self._rule_based_extraction(text)
            
            sop_doc = self._create_sop_document(extracted_data, doc_id, source_info)
            
            return ExtractionResult(
                success=True,
                sop_document=sop_doc,
                warnings=["Used rule-based extraction - results may be less comprehensive"]
            )
            
        except Exception as e:
            logger.error(f"Local extraction error: {e}")
            raise
    
    def _rule_based_extraction(self, text: str) -> Dict[str, Any]:
        """
        Simple rule-based extraction as fallback when LLM models aren't available.
        
        This provides basic structure extraction using text patterns and heuristics.
        """
        import re
        
        # Initialize basic structure
        extracted = {
            "title": "",
            "process_name": "",
            "revision": None,
            "effective_date": None,
            "procedure_steps": [],
            "risks": [],
            "controls": [],
            "roles_responsibilities": [],
            "materials_equipment": [],
            "definitions_glossary": []
        }
        
        lines = text.split('\n')
        
        # Extract title (usually first significant line)
        for line in lines[:10]:
            line = line.strip()
            if len(line) > 10 and not line.startswith(('Page', 'Document', 'Rev')):
                extracted["title"] = line
                extracted["process_name"] = line
                break
        
        # Extract revision information
        revision_pattern = r'(?:Rev|Revision|Version)[\s:]*([A-Za-z0-9.-]+)'
        for line in lines[:20]:
            match = re.search(revision_pattern, line, re.IGNORECASE)
            if match:
                extracted["revision"] = match.group(1)
                break
        
        # Extract procedure steps (numbered items)
        step_pattern = r'^(\d+(?:\.\d+)*)\s*[.-]?\s*(.+)$'
        for line in lines:
            line = line.strip()
            match = re.match(step_pattern, line)
            if match:
                step_id = match.group(1)
                description = match.group(2)
                
                extracted["procedure_steps"].append({
                    "step_id": step_id,
                    "description": description,
                    "responsible_roles": [],
                    "required_equipment": [],
                    "materials": []
                })
        
        # Extract equipment mentions
        equipment_keywords = ['equipment', 'tool', 'machine', 'device', 'instrument']
        for line in lines:
            line_lower = line.lower()
            for keyword in equipment_keywords:
                if keyword in line_lower:
                    # Simple extraction of capitalized words that might be equipment
                    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', line)
                    extracted["materials_equipment"].extend(words)
        
        # Remove duplicates
        extracted["materials_equipment"] = list(set(extracted["materials_equipment"]))
        
        return extracted
    
    def _create_sop_document(self, extracted_data: Dict[str, Any], doc_id: str, source_info: SourceInfo) -> SOPDocument:
        """Create SOPDocument from extracted data with validation."""
        try:
            # Ensure required fields have defaults
            extracted_data.setdefault("doc_id", doc_id)
            extracted_data.setdefault("title", "Untitled SOP")
            extracted_data.setdefault("process_name", "Unknown Process")
            extracted_data["source"] = source_info.model_dump()
            
            # Create and validate SOP document
            sop_doc = SOPDocument(**extracted_data)
            return sop_doc
            
        except ValidationError as e:
            logger.warning(f"Validation errors in extracted data: {e}")
            # Create minimal valid document
            return SOPDocument(
                doc_id=doc_id,
                title=extracted_data.get("title", "Untitled SOP"),
                process_name=extracted_data.get("process_name", "Unknown Process"),
                source=source_info
            )
    
    def _merge_extractions(self, extractions: List[SOPDocument], doc_id: str, source_info: SourceInfo) -> SOPDocument:
        """
        Merge multiple partial extractions into a single comprehensive SOP document.
        
        Args:
            extractions: List of partial SOP documents to merge
            doc_id: Final document ID
            source_info: Source information for the merged document
            
        Returns:
            Merged SOPDocument
        """
        if not extractions:
            raise ValueError("No extractions to merge")
        
        if len(extractions) == 1:
            merged = extractions[0].model_copy()
            merged.doc_id = doc_id
            merged.source = source_info
            return merged
        
        # Start with the first extraction as base
        merged = extractions[0].model_copy()
        merged.doc_id = doc_id
        merged.source = source_info
        
        # Merge data from other extractions
        for extraction in extractions[1:]:
            # Merge procedure steps (maintain order by step_id)
            existing_step_ids = {step.step_id for step in merged.procedure_steps}
            for step in extraction.procedure_steps:
                if step.step_id not in existing_step_ids:
                    merged.procedure_steps.append(step)
            
            # Merge risks (avoid duplicates by risk_id)
            existing_risk_ids = {risk.risk_id for risk in merged.risks}
            for risk in extraction.risks:
                if risk.risk_id not in existing_risk_ids:
                    merged.risks.append(risk)
            
            # Merge controls (avoid duplicates by control_id)
            existing_control_ids = {control.control_id for control in merged.controls}
            for control in extraction.controls:
                if control.control_id not in existing_control_ids:
                    merged.controls.append(control)
            
            # Merge roles (avoid duplicates by role name)
            existing_roles = {role.role for role in merged.roles_responsibilities}
            for role in extraction.roles_responsibilities:
                if role.role not in existing_roles:
                    merged.roles_responsibilities.append(role)
            
            # Merge equipment and materials (avoid duplicates)
            merged.materials_equipment = list(set(
                merged.materials_equipment + extraction.materials_equipment
            ))
            
            # Merge definitions (avoid duplicates by term)
            existing_terms = {defn.term for defn in merged.definitions_glossary}
            for defn in extraction.definitions_glossary:
                if defn.term not in existing_terms:
                    merged.definitions_glossary.append(defn)
            
            # Use more complete metadata if available
            if not merged.title and extraction.title:
                merged.title = extraction.title
            if not merged.revision and extraction.revision:
                merged.revision = extraction.revision
            if not merged.effective_date and extraction.effective_date:
                merged.effective_date = extraction.effective_date
        
        # Sort procedure steps by step_id
        merged.procedure_steps.sort(key=lambda x: self._parse_step_id(x.step_id))
        
        return merged
    
    def _parse_step_id(self, step_id: str) -> Tuple[int, ...]:
        """Parse step ID for sorting (e.g., '1.2.3' -> (1, 2, 3))."""
        try:
            return tuple(int(x) for x in step_id.split('.'))
        except ValueError:
            return (999,)  # Put invalid step IDs at the end
    
    def _validate_extraction(self, sop_document: Optional[SOPDocument]) -> ValidationResult:
        """
        Validate extracted SOP document for completeness and consistency.
        
        Args:
            sop_document: SOP document to validate
            
        Returns:
            ValidationResult with validation status and metrics
        """
        if not sop_document:
            return ValidationResult(
                is_valid=False,
                errors=["No SOP document to validate"],
                completeness_score=0.0
            )
        
        errors = []
        warnings = []
        
        # Check required fields
        if not sop_document.title or sop_document.title == "Untitled SOP":
            warnings.append("Document title is missing or generic")
        
        if not sop_document.procedure_steps:
            errors.append("No procedure steps found")
        
        # Check step ID consistency
        step_ids = [step.step_id for step in sop_document.procedure_steps]
        if len(step_ids) != len(set(step_ids)):
            errors.append("Duplicate step IDs found")
        
        # Check referential integrity
        all_step_ids = set(step_ids)
        for risk in sop_document.risks:
            for step_id in risk.affected_steps:
                if step_id not in all_step_ids:
                    warnings.append(f"Risk {risk.risk_id} references non-existent step {step_id}")
        
        for control in sop_document.controls:
            for step_id in control.applicable_steps:
                if step_id not in all_step_ids:
                    warnings.append(f"Control {control.control_id} references non-existent step {step_id}")
        
        # Calculate completeness score
        completeness_factors = [
            bool(sop_document.title and sop_document.title != "Untitled SOP"),
            bool(sop_document.procedure_steps),
            bool(sop_document.roles_responsibilities),
            bool(sop_document.materials_equipment),
            bool(sop_document.risks),
            bool(sop_document.controls),
            bool(sop_document.revision),
            bool(sop_document.effective_date)
        ]
        completeness_score = sum(completeness_factors) / len(completeness_factors)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            completeness_score=completeness_score
        )
    
    def _get_sop_schema(self) -> Dict[str, Any]:
        """Get JSON schema for SOP document structure."""
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "process_name": {"type": "string"},
                "revision": {"type": ["string", "null"]},
                "effective_date": {"type": ["string", "null"]},
                "owner_role": {"type": ["string", "null"]},
                "scope": {"type": ["string", "null"]},
                "definitions_glossary": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "term": {"type": "string"},
                            "definition": {"type": "string"},
                            "category": {"type": ["string", "null"]}
                        },
                        "required": ["term", "definition"]
                    }
                },
                "preconditions": {"type": "array", "items": {"type": "string"}},
                "materials_equipment": {"type": "array", "items": {"type": "string"}},
                "roles_responsibilities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "responsibilities": {"type": "array", "items": {"type": "string"}},
                            "qualifications": {"type": ["array", "null"], "items": {"type": "string"}},
                            "authority_level": {"type": ["string", "null"]}
                        },
                        "required": ["role", "responsibilities"]
                    }
                },
                "procedure_steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_id": {"type": "string"},
                            "title": {"type": ["string", "null"]},
                            "description": {"type": "string"},
                            "step_type": {"type": ["string", "null"]},
                            "responsible_roles": {"type": "array", "items": {"type": "string"}},
                            "required_equipment": {"type": "array", "items": {"type": "string"}},
                            "materials": {"type": "array", "items": {"type": "string"}},
                            "duration_minutes": {"type": ["integer", "null"]},
                            "prerequisites": {"type": "array", "items": {"type": "string"}},
                            "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                            "safety_notes": {"type": "array", "items": {"type": "string"}},
                            "quality_checkpoints": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["step_id", "description"]
                    }
                },
                "risks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "risk_id": {"type": "string"},
                            "description": {"type": "string"},
                            "category": {"type": "string"},
                            "probability": {"type": ["string", "null"]},
                            "severity": {"type": ["string", "null"]},
                            "overall_rating": {"type": ["string", "null"]},
                            "affected_steps": {"type": "array", "items": {"type": "string"}},
                            "potential_consequences": {"type": "array", "items": {"type": "string"}},
                            "triggers": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["risk_id", "description", "category"]
                    }
                },
                "controls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "control_id": {"type": "string"},
                            "description": {"type": "string"},
                            "control_type": {"type": "string"},
                            "effectiveness": {"type": ["string", "null"]},
                            "applicable_risks": {"type": "array", "items": {"type": "string"}},
                            "applicable_steps": {"type": "array", "items": {"type": "string"}},
                            "responsible_roles": {"type": "array", "items": {"type": "string"}},
                            "verification_method": {"type": ["string", "null"]},
                            "frequency": {"type": ["string", "null"]}
                        },
                        "required": ["control_id", "description", "control_type"]
                    }
                },
                "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                "compliance_refs": {"type": "array", "items": {"type": "string"}},
                "attachments_refs": {"type": "array", "items": {"type": "string"}},
                "change_log": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "version": {"type": "string"},
                            "date": {"type": "string"},
                            "author": {"type": "string"},
                            "description": {"type": "string"},
                            "approval_status": {"type": ["string", "null"]}
                        },
                        "required": ["version", "date", "author", "description"]
                    }
                }
            },
            "required": ["title", "process_name"]
        }