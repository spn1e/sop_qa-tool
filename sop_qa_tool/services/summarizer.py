"""
Document Summarizer Service for SOP documents.

This service creates structured summaries of SOP documents that complement
the detailed ontology extraction. It provides high-level overviews and
key insights for quick document understanding.

Requirements: 2.1, 2.2, 2.3
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..models.sop_models import SOPDocument, DocumentChunk
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class SOPSummarizer:
    """
    Creates structured summaries of SOP documents for quick understanding
    and enhanced retrieval context.
    """
    
    def __init__(self):
        self.settings = get_settings()
    
    def create_document_summary(self, sop_document: SOPDocument) -> Dict[str, Any]:
        """
        Create a comprehensive summary of an SOP document.
        
        Args:
            sop_document: The SOP document to summarize
            
        Returns:
            Dictionary containing structured summary information
        """
        try:
            summary = {
                "doc_id": sop_document.doc_id,
                "title": sop_document.title,
                "process_name": sop_document.process_name,
                "revision": sop_document.revision,
                "effective_date": sop_document.effective_date.isoformat() if sop_document.effective_date else None,
                "summary_created": datetime.utcnow().isoformat(),
                
                # High-level metrics
                "metrics": self._calculate_metrics(sop_document),
                
                # Key components summary
                "overview": self._create_overview(sop_document),
                
                # Critical information
                "critical_info": self._extract_critical_info(sop_document),
                
                # Quick reference
                "quick_reference": self._create_quick_reference(sop_document)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to create summary for {sop_document.doc_id}: {e}")
            return self._create_minimal_summary(sop_document)
    
    def create_chunk_summary(self, chunk: DocumentChunk, context: Optional[SOPDocument] = None) -> Dict[str, Any]:
        """
        Create a summary for a specific document chunk.
        
        Args:
            chunk: The document chunk to summarize
            context: Optional full SOP document for additional context
            
        Returns:
            Dictionary containing chunk summary information
        """
        try:
            summary = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "page_no": chunk.page_no,
                "heading_path": chunk.heading_path,
                
                # Content analysis
                "content_type": self._identify_content_type(chunk.chunk_text),
                "key_topics": self._extract_key_topics(chunk.chunk_text),
                "mentioned_entities": {
                    "roles": chunk.roles,
                    "equipment": chunk.equipment,
                    "steps": chunk.step_ids,
                    "risks": chunk.risk_ids,
                    "controls": chunk.control_ids
                },
                
                # Text statistics
                "text_stats": {
                    "word_count": len(chunk.chunk_text.split()),
                    "char_count": len(chunk.chunk_text),
                    "sentence_count": len([s for s in chunk.chunk_text.split('.') if s.strip()])
                }
            }
            
            # Add context-specific information if available
            if context:
                summary["context_info"] = self._add_context_info(chunk, context)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to create chunk summary for {chunk.chunk_id}: {e}")
            return {"chunk_id": chunk.chunk_id, "error": str(e)}
    
    def _calculate_metrics(self, sop_document: SOPDocument) -> Dict[str, Any]:
        """Calculate key metrics for the SOP document."""
        return {
            "total_steps": len(sop_document.procedure_steps),
            "total_risks": len(sop_document.risks),
            "total_controls": len(sop_document.controls),
            "total_roles": len(sop_document.roles_responsibilities),
            "total_equipment": len(sop_document.materials_equipment),
            "total_definitions": len(sop_document.definitions_glossary),
            
            # Risk analysis
            "high_priority_risks": len([r for r in sop_document.risks 
                                      if r.overall_rating in ["high", "critical"]]),
            "risk_categories": list(set(r.category.value for r in sop_document.risks)),
            
            # Control analysis
            "control_types": list(set(c.control_type.value for c in sop_document.controls)),
            
            # Step analysis
            "step_types": list(set(s.step_type.value for s in sop_document.procedure_steps 
                                 if s.step_type)),
            "steps_with_safety_notes": len([s for s in sop_document.procedure_steps 
                                          if s.safety_notes]),
            "steps_with_quality_checks": len([s for s in sop_document.procedure_steps 
                                            if s.quality_checkpoints])
        }
    
    def _create_overview(self, sop_document: SOPDocument) -> Dict[str, Any]:
        """Create a high-level overview of the SOP."""
        return {
            "purpose": sop_document.scope or f"Standard operating procedure for {sop_document.process_name}",
            "owner": sop_document.owner_role,
            "complexity_level": self._assess_complexity(sop_document),
            "main_phases": self._identify_main_phases(sop_document),
            "key_roles": [role.role for role in sop_document.roles_responsibilities[:5]],  # Top 5 roles
            "critical_equipment": self._identify_critical_equipment(sop_document),
            "safety_focus": len([s for s in sop_document.procedure_steps if s.safety_notes]) > 0
        }
    
    def _extract_critical_info(self, sop_document: SOPDocument) -> Dict[str, Any]:
        """Extract critical information that users need to know immediately."""
        critical_risks = [
            {
                "risk_id": risk.risk_id,
                "description": risk.description,
                "category": risk.category.value,
                "rating": risk.overall_rating.value if risk.overall_rating else "unknown"
            }
            for risk in sop_document.risks
            if risk.overall_rating and risk.overall_rating.value in ["high", "critical"]
        ]
        
        safety_steps = [
            {
                "step_id": step.step_id,
                "title": step.title or step.description[:50] + "...",
                "safety_notes": step.safety_notes
            }
            for step in sop_document.procedure_steps
            if step.safety_notes
        ]
        
        return {
            "critical_risks": critical_risks,
            "safety_critical_steps": safety_steps,
            "compliance_requirements": sop_document.compliance_refs,
            "prerequisites": sop_document.preconditions,
            "required_qualifications": self._extract_qualifications(sop_document)
        }
    
    def _create_quick_reference(self, sop_document: SOPDocument) -> Dict[str, Any]:
        """Create quick reference information for easy lookup."""
        return {
            "step_sequence": [
                {
                    "step_id": step.step_id,
                    "title": step.title or step.description[:30] + "...",
                    "responsible_role": step.responsible_roles[0] if step.responsible_roles else "Unassigned",
                    "duration": step.duration_minutes
                }
                for step in sop_document.procedure_steps[:10]  # First 10 steps
            ],
            "equipment_checklist": sop_document.materials_equipment,
            "role_assignments": {
                role.role: len(role.responsibilities)
                for role in sop_document.roles_responsibilities
            },
            "key_definitions": [
                {"term": defn.term, "definition": defn.definition[:100] + "..."}
                for defn in sop_document.definitions_glossary[:5]  # Top 5 definitions
            ]
        }
    
    def _assess_complexity(self, sop_document: SOPDocument) -> str:
        """Assess the complexity level of the SOP."""
        complexity_score = 0
        
        # Factor in number of steps
        if len(sop_document.procedure_steps) > 20:
            complexity_score += 2
        elif len(sop_document.procedure_steps) > 10:
            complexity_score += 1
        
        # Factor in number of roles
        if len(sop_document.roles_responsibilities) > 5:
            complexity_score += 1
        
        # Factor in number of risks
        if len(sop_document.risks) > 10:
            complexity_score += 1
        
        # Factor in nested steps
        nested_steps = sum(1 for step in sop_document.procedure_steps 
                          if '.' in step.step_id and step.step_id.count('.') > 1)
        if nested_steps > 5:
            complexity_score += 1
        
        if complexity_score >= 4:
            return "high"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _identify_main_phases(self, sop_document: SOPDocument) -> List[str]:
        """Identify main phases or sections of the procedure."""
        phases = []
        
        # Look for top-level steps (no decimal points)
        top_level_steps = [step for step in sop_document.procedure_steps 
                          if '.' not in step.step_id]
        
        for step in top_level_steps:
            phase_name = step.title or step.description[:50]
            phases.append(phase_name)
        
        # If no clear phases, group by step type
        if not phases:
            step_types = set(step.step_type.value for step in sop_document.procedure_steps 
                           if step.step_type)
            phases = list(step_types)
        
        return phases[:5]  # Limit to 5 main phases
    
    def _identify_critical_equipment(self, sop_document: SOPDocument) -> List[str]:
        """Identify equipment that appears most frequently or is marked as critical."""
        equipment_mentions = {}
        
        # Count equipment mentions across steps
        for step in sop_document.procedure_steps:
            for equipment in step.required_equipment:
                equipment_mentions[equipment] = equipment_mentions.get(equipment, 0) + 1
        
        # Sort by frequency and return top items
        sorted_equipment = sorted(equipment_mentions.items(), key=lambda x: x[1], reverse=True)
        return [eq[0] for eq in sorted_equipment[:5]]
    
    def _extract_qualifications(self, sop_document: SOPDocument) -> List[str]:
        """Extract required qualifications from roles."""
        qualifications = []
        for role in sop_document.roles_responsibilities:
            if role.qualifications:
                qualifications.extend(role.qualifications)
        return list(set(qualifications))  # Remove duplicates
    
    def _identify_content_type(self, text: str) -> str:
        """Identify the type of content in a text chunk."""
        text_lower = text.lower()
        
        # Check for different content patterns
        if any(word in text_lower for word in ['step', 'procedure', 'perform', 'execute']):
            return "procedure_step"
        elif any(word in text_lower for word in ['risk', 'hazard', 'danger', 'caution']):
            return "risk_information"
        elif any(word in text_lower for word in ['control', 'mitigation', 'prevention']):
            return "control_measure"
        elif any(word in text_lower for word in ['role', 'responsibility', 'responsible']):
            return "role_definition"
        elif any(word in text_lower for word in ['equipment', 'tool', 'material', 'supply']):
            return "equipment_list"
        elif any(word in text_lower for word in ['definition', 'glossary', 'term', 'means']):
            return "definition"
        else:
            return "general_content"
    
    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics or themes from text."""
        # Simple keyword extraction based on common SOP terms
        keywords = [
            'safety', 'quality', 'procedure', 'equipment', 'material', 'inspection',
            'verification', 'control', 'risk', 'hazard', 'training', 'qualification',
            'documentation', 'record', 'approval', 'review', 'maintenance', 'cleaning',
            'calibration', 'testing', 'monitoring', 'compliance', 'standard', 'regulation'
        ]
        
        text_lower = text.lower()
        found_topics = [keyword for keyword in keywords if keyword in text_lower]
        
        return found_topics[:5]  # Limit to top 5 topics
    
    def _add_context_info(self, chunk: DocumentChunk, context: SOPDocument) -> Dict[str, Any]:
        """Add contextual information from the full SOP document."""
        context_info = {}
        
        # Find related steps
        if chunk.step_ids:
            related_steps = [step for step in context.procedure_steps 
                           if step.step_id in chunk.step_ids]
            context_info["related_steps"] = [
                {"step_id": step.step_id, "title": step.title}
                for step in related_steps
            ]
        
        # Find related risks
        if chunk.risk_ids:
            related_risks = [risk for risk in context.risks 
                           if risk.risk_id in chunk.risk_ids]
            context_info["related_risks"] = [
                {"risk_id": risk.risk_id, "category": risk.category.value}
                for risk in related_risks
            ]
        
        # Find related controls
        if chunk.control_ids:
            related_controls = [control for control in context.controls 
                              if control.control_id in chunk.control_ids]
            context_info["related_controls"] = [
                {"control_id": control.control_id, "type": control.control_type.value}
                for control in related_controls
            ]
        
        return context_info
    
    def _create_minimal_summary(self, sop_document: SOPDocument) -> Dict[str, Any]:
        """Create a minimal summary when full processing fails."""
        return {
            "doc_id": sop_document.doc_id,
            "title": sop_document.title,
            "process_name": sop_document.process_name,
            "error": "Failed to create full summary",
            "basic_metrics": {
                "total_steps": len(sop_document.procedure_steps),
                "total_risks": len(sop_document.risks),
                "total_controls": len(sop_document.controls)
            }
        }