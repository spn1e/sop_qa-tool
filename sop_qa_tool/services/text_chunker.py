"""
Text Chunking and Processing Service

Implements recursive text splitter with configurable chunk size and overlap,
heading extraction, structure preservation, and metadata enrichment for chunks.
Provides deterministic chunking with consistent results for same input.
"""

import hashlib
import logging
import re
import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path

from ..config.settings import get_settings
from ..models.sop_models import DocumentChunk, SOPDocument


logger = logging.getLogger(__name__)


@dataclass
class HeadingInfo:
    """Information about a heading in the document"""
    level: int  # 1 for H1, 2 for H2, etc.
    text: str
    start_pos: int
    end_pos: int
    path: str  # Hierarchical path like "3. Process > 3.2 Setup"


@dataclass
class ChunkMetadata:
    """Metadata extracted for a chunk"""
    step_ids: List[str]
    risk_ids: List[str]
    control_ids: List[str]
    roles: List[str]
    equipment: List[str]
    page_no: Optional[int] = None
    heading_path: Optional[str] = None


class TextChunker:
    """
    Text chunking service with structure preservation and metadata enrichment.
    
    Implements recursive text splitting with configurable parameters and
    deterministic results for consistent chunking of the same input.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.chunk_size = self.settings.chunk_size
        self.chunk_overlap = self.settings.chunk_overlap
        
        # Patterns for extracting metadata
        self._step_pattern = re.compile(r'\b(?:step\s+)?(\d+(?:\.\d+)*)\b', re.IGNORECASE)
        self._risk_pattern = re.compile(r'\b(?:risk\s+)?[rR]-?(\d+)\b')
        self._control_pattern = re.compile(r'\b(?:control\s+)?[cC]-?(\d+)\b')
        
        # Common role patterns in manufacturing SOPs
        self._role_patterns = [
            r'\b(?:operator|technician|supervisor|manager|inspector|auditor)\b',
            r'\b(?:QA|QC|quality\s+(?:assurance|control))\b',
            r'\b(?:line\s+(?:supervisor|lead|operator))\b',
            r'\b(?:maintenance\s+(?:technician|engineer))\b',
            r'\b(?:safety\s+(?:officer|coordinator))\b',
            r'\b(?:production\s+(?:manager|supervisor|lead))\b'
        ]
        
        # Common equipment patterns
        self._equipment_patterns = [
            r'\b(?:filler|mixer|conveyor|pump|valve|sensor|probe)\b',
            r'\b(?:machine|equipment|device|instrument|tool)\s+\w+',
            r'\b\w+[-_]\d+\b',  # Equipment IDs like "FILLER-01"
            r'\b(?:station|line|cell)\s+\d+\b'
        ]
        
        # Heading patterns for different levels
        self._heading_patterns = [
            re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE),  # Markdown headings
            re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$', re.MULTILINE),  # Numbered headings
            re.compile(r'^([A-Z][A-Z\s]+)$', re.MULTILINE),  # ALL CAPS headings
            re.compile(r'^(.+)\n[=-]{3,}$', re.MULTILINE)  # Underlined headings
        ]
    
    def chunk_document(
        self, 
        text: str, 
        doc_id: str,
        sop_document: Optional[SOPDocument] = None,
        preserve_structure: bool = True
    ) -> List[DocumentChunk]:
        """
        Chunk document text with structure preservation and metadata enrichment.
        
        Args:
            text: Raw document text
            doc_id: Document identifier
            sop_document: Optional structured SOP document for enhanced metadata
            preserve_structure: Whether to preserve document structure in chunks
            
        Returns:
            List of DocumentChunk objects with metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Chunking document {doc_id}: {len(text)} chars")
            
            # Extract headings for structure preservation
            headings = self._extract_headings(text) if preserve_structure else []
            
            # Split text into chunks
            raw_chunks = self._recursive_split(text, self.chunk_size, self.chunk_overlap)
            
            # Create DocumentChunk objects with metadata
            chunks = []
            for i, chunk_text in enumerate(raw_chunks):
                # Generate deterministic chunk ID
                chunk_id = self._generate_chunk_id(doc_id, i, chunk_text)
                
                # Find chunk position in original text
                chunk_start = text.find(chunk_text)
                
                # Extract metadata for this chunk
                metadata = self._extract_chunk_metadata(
                    chunk_text, 
                    chunk_start, 
                    headings, 
                    sop_document
                )
                
                # Create chunk object
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    chunk_text=chunk_text,
                    chunk_index=i,
                    page_no=metadata.page_no,
                    heading_path=metadata.heading_path,
                    step_ids=metadata.step_ids,
                    risk_ids=metadata.risk_ids,
                    control_ids=metadata.control_ids,
                    roles=metadata.roles,
                    equipment=metadata.equipment
                )
                
                chunks.append(chunk)
            
            processing_time = time.time() - start_time
            logger.info(
                f"Document chunking complete: {len(chunks)} chunks in {processing_time:.2f}s"
            )
            
            return chunks
            
        except Exception as e:
            logger.error(f"Document chunking failed for {doc_id}: {e}")
            raise
    
    def _recursive_split(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int,
        separators: Optional[List[str]] = None
    ) -> List[str]:
        """
        Recursively split text into chunks with overlap.
        
        Args:
            text: Text to split
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters
            separators: List of separators to try (in order of preference)
            
        Returns:
            List of text chunks
        """
        if separators is None:
            # Default separators in order of preference
            separators = [
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence endings
                "! ",    # Exclamation sentences
                "? ",    # Question sentences
                "; ",    # Semicolons
                ", ",    # Commas
                " ",     # Spaces
                ""       # Character-level split (last resort)
            ]
        
        chunks = []
        
        # If text is small enough, return as single chunk
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []
        
        # Try each separator
        for separator in separators:
            if separator == "":
                # Character-level split as last resort
                chunks = self._split_by_characters(text, chunk_size, overlap)
                break
            
            # Split by current separator
            splits = text.split(separator)
            
            # If we got meaningful splits, process them
            if len(splits) > 1:
                chunks = self._merge_splits(splits, separator, chunk_size, overlap)
                break
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _split_by_characters(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text by characters when no good separators are found"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Don't split in the middle of a word if possible
            if end < len(text) and text[end] != ' ':
                # Look for the nearest space before the end
                space_pos = text.rfind(' ', start, end)
                if space_pos > start:
                    end = space_pos
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - overlap)
        
        return chunks
    
    def _merge_splits(
        self, 
        splits: List[str], 
        separator: str, 
        chunk_size: int, 
        overlap: int
    ) -> List[str]:
        """
        Merge splits into chunks of appropriate size with overlap.
        
        Args:
            splits: List of text splits
            separator: Separator used for splitting
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            
        Returns:
            List of merged chunks
        """
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # Calculate size if we add this split
            potential_chunk = current_chunk
            if potential_chunk:
                potential_chunk += separator + split
            else:
                potential_chunk = split
            
            # If adding this split would exceed chunk size
            if len(potential_chunk) > chunk_size and current_chunk:
                # Save current chunk and start new one
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    # Take last part of current chunk as overlap
                    overlap_text = current_chunk[-overlap:].strip()
                    current_chunk = overlap_text + separator + split if overlap_text else split
                else:
                    current_chunk = split
            else:
                # Add split to current chunk
                current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _extract_headings(self, text: str) -> List[HeadingInfo]:
        """
        Extract headings from text to preserve document structure.
        
        Args:
            text: Document text
            
        Returns:
            List of HeadingInfo objects
        """
        headings = []
        
        # Try different heading patterns
        for pattern in self._heading_patterns:
            matches = pattern.finditer(text)
            
            for match in matches:
                heading_text = match.group(1).strip()
                start_pos = match.start()
                end_pos = match.end()
                
                # Determine heading level
                level = self._determine_heading_level(match, heading_text)
                
                # Build hierarchical path
                path = self._build_heading_path(headings, level, heading_text)
                
                heading = HeadingInfo(
                    level=level,
                    text=heading_text,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    path=path
                )
                
                headings.append(heading)
        
        # Sort headings by position
        headings.sort(key=lambda h: h.start_pos)
        
        # Remove duplicates and overlaps
        headings = self._deduplicate_headings(headings)
        
        return headings
    
    def _determine_heading_level(self, match: re.Match, heading_text: str) -> int:
        """Determine the level of a heading based on the match pattern"""
        full_match = match.group(0)
        
        # Markdown headings
        if full_match.startswith('#'):
            return len(full_match) - len(full_match.lstrip('#'))
        
        # Numbered headings (count dots)
        if re.match(r'^\d+(?:\.\d+)*', full_match):
            return full_match.count('.') + 1
        
        # ALL CAPS headings are usually level 1
        if heading_text.isupper():
            return 1
        
        # Underlined headings
        if '\n' in full_match and re.search(r'[=-]{3,}$', full_match):
            return 1 if '=' in full_match else 2
        
        return 1  # Default level
    
    def _build_heading_path(
        self, 
        existing_headings: List[HeadingInfo], 
        level: int, 
        heading_text: str
    ) -> str:
        """Build hierarchical path for a heading"""
        # Find parent headings
        parent_headings = []
        for heading in reversed(existing_headings):
            if heading.level < level:
                parent_headings.insert(0, heading)
                level = heading.level
            if level == 1:
                break
        
        # Build path
        path_parts = [h.text for h in parent_headings] + [heading_text]
        return " > ".join(path_parts)
    
    def _deduplicate_headings(self, headings: List[HeadingInfo]) -> List[HeadingInfo]:
        """Remove duplicate and overlapping headings"""
        if not headings:
            return []
        
        deduplicated = []
        last_end = -1
        
        for heading in headings:
            # Skip if this heading overlaps with the previous one
            if heading.start_pos < last_end:
                continue
            
            # Skip if this is a duplicate of the previous heading
            if (deduplicated and 
                deduplicated[-1].text.lower() == heading.text.lower() and
                abs(deduplicated[-1].start_pos - heading.start_pos) < 100):
                continue
            
            deduplicated.append(heading)
            last_end = heading.end_pos
        
        return deduplicated
    
    def _extract_chunk_metadata(
        self,
        chunk_text: str,
        chunk_start: int,
        headings: List[HeadingInfo],
        sop_document: Optional[SOPDocument] = None
    ) -> ChunkMetadata:
        """
        Extract metadata for a text chunk.
        
        Args:
            chunk_text: Text content of the chunk
            chunk_start: Starting position of chunk in original text
            headings: List of document headings
            sop_document: Optional structured SOP document
            
        Returns:
            ChunkMetadata object
        """
        # Find relevant heading for this chunk
        heading_path = self._find_chunk_heading(chunk_start, headings)
        
        # Extract step IDs
        step_ids = self._extract_step_ids(chunk_text)
        
        # Extract risk IDs
        risk_ids = self._extract_risk_ids(chunk_text)
        
        # Extract control IDs
        control_ids = self._extract_control_ids(chunk_text)
        
        # Extract roles
        roles = self._extract_roles(chunk_text, sop_document)
        
        # Extract equipment
        equipment = self._extract_equipment(chunk_text, sop_document)
        
        # Extract page number (if available in text)
        page_no = self._extract_page_number(chunk_text)
        
        return ChunkMetadata(
            step_ids=step_ids,
            risk_ids=risk_ids,
            control_ids=control_ids,
            roles=roles,
            equipment=equipment,
            page_no=page_no,
            heading_path=heading_path
        )
    
    def _find_chunk_heading(
        self, 
        chunk_start: int, 
        headings: List[HeadingInfo]
    ) -> Optional[str]:
        """Find the most relevant heading for a chunk position"""
        relevant_heading = None
        
        for heading in headings:
            if heading.start_pos <= chunk_start:
                relevant_heading = heading
            else:
                break
        
        return relevant_heading.path if relevant_heading else None
    
    def _extract_step_ids(self, text: str) -> List[str]:
        """Extract step IDs from text"""
        matches = self._step_pattern.findall(text)
        return list(set(matches))  # Remove duplicates
    
    def _extract_risk_ids(self, text: str) -> List[str]:
        """Extract risk IDs from text"""
        matches = self._risk_pattern.findall(text)
        return [f"R-{match}" for match in set(matches)]
    
    def _extract_control_ids(self, text: str) -> List[str]:
        """Extract control IDs from text"""
        matches = self._control_pattern.findall(text)
        return [f"C-{match}" for match in set(matches)]
    
    def _extract_roles(
        self, 
        text: str, 
        sop_document: Optional[SOPDocument] = None
    ) -> List[str]:
        """Extract role mentions from text"""
        roles = set()
        
        # Use patterns to find roles
        for pattern in self._role_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            roles.update(match.lower() for match in matches)
        
        # If we have structured SOP data, also check against known roles
        if sop_document:
            known_roles = {role.role.lower() for role in sop_document.roles_responsibilities}
            for role in known_roles:
                if role in text.lower():
                    roles.add(role)
        
        return list(roles)
    
    def _extract_equipment(
        self, 
        text: str, 
        sop_document: Optional[SOPDocument] = None
    ) -> List[str]:
        """Extract equipment mentions from text"""
        equipment = set()
        
        # Use patterns to find equipment
        for pattern in self._equipment_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            equipment.update(match.lower() for match in matches)
        
        # If we have structured SOP data, check against known equipment
        if sop_document:
            known_equipment = set()
            for step in sop_document.procedure_steps:
                known_equipment.update(eq.lower() for eq in step.required_equipment)
            
            # Also check materials_equipment list
            known_equipment.update(eq.lower() for eq in sop_document.materials_equipment)
            
            for eq in known_equipment:
                if eq in text.lower():
                    equipment.add(eq)
        
        return list(equipment)
    
    def _extract_page_number(self, text: str) -> Optional[int]:
        """Extract page number from text if present"""
        # Look for page indicators
        page_patterns = [
            r'page\s+(\d+)',
            r'p\.?\s*(\d+)',
            r'pg\.?\s*(\d+)'
        ]
        
        for pattern in page_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def _generate_chunk_id(self, doc_id: str, chunk_index: int, chunk_text: str) -> str:
        """
        Generate deterministic chunk ID based on content.
        
        This ensures consistent chunk IDs for the same input, which is important
        for reproducible results and avoiding duplicate processing.
        """
        # Create hash of chunk content for deterministic ID
        content_hash = hashlib.md5(chunk_text.encode('utf-8')).hexdigest()[:8]
        return f"{doc_id}_chunk_{chunk_index:03d}_{content_hash}"
    
    def validate_chunks(self, chunks: List[DocumentChunk]) -> Dict[str, any]:
        """
        Validate chunk consistency and quality.
        
        Args:
            chunks: List of document chunks to validate
            
        Returns:
            Dictionary with validation results
        """
        if not chunks:
            return {
                'valid': False,
                'errors': ['No chunks provided'],
                'warnings': [],
                'stats': {}
            }
        
        errors = []
        warnings = []
        
        # Check for duplicate chunk IDs
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            errors.append("Duplicate chunk IDs found")
        
        # Check chunk sizes
        chunk_sizes = [len(chunk.chunk_text) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        
        if avg_size > self.chunk_size * 1.5:
            warnings.append(f"Average chunk size ({avg_size:.0f}) exceeds target by 50%")
        
        # Check for empty chunks
        empty_chunks = [chunk for chunk in chunks if not chunk.chunk_text.strip()]
        if empty_chunks:
            warnings.append(f"Found {len(empty_chunks)} empty chunks")
        
        # Check chunk indices are sequential
        indices = [chunk.chunk_index for chunk in chunks]
        expected_indices = list(range(len(chunks)))
        if indices != expected_indices:
            errors.append("Chunk indices are not sequential")
        
        # Calculate statistics
        stats = {
            'total_chunks': len(chunks),
            'avg_chunk_size': avg_size,
            'min_chunk_size': min(chunk_sizes) if chunk_sizes else 0,
            'max_chunk_size': max(chunk_sizes) if chunk_sizes else 0,
            'total_characters': sum(chunk_sizes),
            'chunks_with_metadata': len([c for c in chunks if c.step_ids or c.roles or c.equipment])
        }
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'stats': stats
        }