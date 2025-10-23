"""
SOP Card Display Components

Structured display components for SOP ontology information with rich formatting.

Requirements: 5.2, 4.2, 4.3
"""

import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


def render_sop_card(sop_data: Dict[str, Any], expanded: bool = False):
    """
    Render a structured SOP card with ontology information.
    
    Args:
        sop_data: SOP document data with ontology information
        expanded: Whether to show the card expanded by default
    """
    if not sop_data:
        st.warning("No SOP data available")
        return
    
    # Card header
    title = sop_data.get("title", "Unknown SOP")
    doc_id = sop_data.get("doc_id", "N/A")
    
    with st.expander(f"📋 {title}", expanded=expanded):
        # Basic information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Document Information**")
            st.write(f"**ID:** {doc_id}")
            st.write(f"**Process:** {sop_data.get('process_name', 'N/A')}")
            st.write(f"**Revision:** {sop_data.get('revision', 'N/A')}")
            st.write(f"**Effective Date:** {sop_data.get('effective_date', 'N/A')}")
        
        with col2:
            st.markdown("**Ownership & Scope**")
            st.write(f"**Owner:** {sop_data.get('owner_role', 'N/A')}")
            scope = sop_data.get('scope', 'N/A')
            if len(scope) > 100:
                scope = scope[:100] + "..."
            st.write(f"**Scope:** {scope}")
        
        # Tabs for detailed information
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔧 Procedure Steps", 
            "⚠️ Risks & Controls", 
            "👥 Roles", 
            "🛠️ Equipment", 
            "📊 Metadata"
        ])
        
        with tab1:
            render_procedure_steps(sop_data.get("procedure_steps", []))
        
        with tab2:
            render_risks_and_controls(
                sop_data.get("risks", []), 
                sop_data.get("controls", [])
            )
        
        with tab3:
            render_roles_responsibilities(sop_data.get("roles_responsibilities", []))
        
        with tab4:
            render_equipment_materials(sop_data.get("materials_equipment", []))
        
        with tab5:
            render_metadata(sop_data)


def render_procedure_steps(steps: List[Dict[str, Any]]):
    """Render procedure steps in a structured format"""
    if not steps:
        st.info("No procedure steps available")
        return
    
    st.markdown("### Procedure Steps")
    
    for i, step in enumerate(steps, 1):
        step_id = step.get("step_id", f"Step {i}")
        description = step.get("description", "No description")
        
        with st.container():
            # Step header
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{step_id}:** {description}")
            with col2:
                if step.get("critical", False):
                    st.error("🚨 Critical")
                elif step.get("safety_critical", False):
                    st.warning("⚠️ Safety")
            
            # Step details
            if step.get("inputs"):
                st.markdown("**Inputs:** " + ", ".join(step["inputs"]))
            
            if step.get("outputs"):
                st.markdown("**Outputs:** " + ", ".join(step["outputs"]))
            
            if step.get("tools_required"):
                st.markdown("**Tools:** " + ", ".join(step["tools_required"]))
            
            if step.get("duration_minutes"):
                st.markdown(f"**Duration:** {step['duration_minutes']} minutes")
            
            if step.get("verification_method"):
                st.markdown(f"**Verification:** {step['verification_method']}")
            
            # Associated risks and controls
            if step.get("associated_risks"):
                with st.expander("⚠️ Associated Risks"):
                    for risk_id in step["associated_risks"]:
                        st.write(f"- {risk_id}")
            
            if step.get("associated_controls"):
                with st.expander("🛡️ Associated Controls"):
                    for control_id in step["associated_controls"]:
                        st.write(f"- {control_id}")
            
            st.divider()


def render_risks_and_controls(risks: List[Dict[str, Any]], controls: List[Dict[str, Any]]):
    """Render risks and controls information"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚠️ Risks")
        if risks:
            for risk in risks:
                risk_id = risk.get("risk_id", "Unknown")
                description = risk.get("description", "No description")
                severity = risk.get("severity", "Unknown")
                probability = risk.get("probability", "Unknown")
                
                # Risk severity color coding
                if severity.lower() in ["high", "critical"]:
                    severity_color = "🔴"
                elif severity.lower() == "medium":
                    severity_color = "🟡"
                else:
                    severity_color = "🟢"
                
                with st.container():
                    st.markdown(f"**{risk_id}** {severity_color}")
                    st.write(description)
                    st.caption(f"Severity: {severity} | Probability: {probability}")
                    
                    if risk.get("mitigation_steps"):
                        with st.expander("Mitigation Steps"):
                            for step in risk["mitigation_steps"]:
                                st.write(f"- {step}")
                    
                    st.divider()
        else:
            st.info("No risks identified")
    
    with col2:
        st.markdown("### 🛡️ Controls")
        if controls:
            for control in controls:
                control_id = control.get("control_id", "Unknown")
                description = control.get("description", "No description")
                control_type = control.get("control_type", "Unknown")
                
                # Control type icon
                type_icons = {
                    "preventive": "🚫",
                    "detective": "🔍",
                    "corrective": "🔧",
                    "administrative": "📋"
                }
                type_icon = type_icons.get(control_type.lower(), "🛡️")
                
                with st.container():
                    st.markdown(f"**{control_id}** {type_icon}")
                    st.write(description)
                    st.caption(f"Type: {control_type}")
                    
                    if control.get("frequency"):
                        st.caption(f"Frequency: {control['frequency']}")
                    
                    if control.get("responsible_role"):
                        st.caption(f"Responsible: {control['responsible_role']}")
                    
                    st.divider()
        else:
            st.info("No controls defined")


def render_roles_responsibilities(roles: List[Dict[str, Any]]):
    """Render roles and responsibilities"""
    if not roles:
        st.info("No roles and responsibilities defined")
        return
    
    st.markdown("### 👥 Roles & Responsibilities")
    
    for role in roles:
        role_name = role.get("role_name", "Unknown Role")
        responsibilities = role.get("responsibilities", [])
        qualifications = role.get("qualifications", [])
        
        with st.container():
            st.markdown(f"**{role_name}**")
            
            if responsibilities:
                st.markdown("**Responsibilities:**")
                for resp in responsibilities:
                    st.write(f"- {resp}")
            
            if qualifications:
                st.markdown("**Qualifications:**")
                for qual in qualifications:
                    st.write(f"- {qual}")
            
            if role.get("training_required"):
                st.markdown(f"**Training Required:** {role['training_required']}")
            
            st.divider()


def render_equipment_materials(equipment: List[str]):
    """Render equipment and materials list"""
    if not equipment:
        st.info("No equipment or materials specified")
        return
    
    st.markdown("### 🛠️ Equipment & Materials")
    
    # Group equipment by type if possible
    equipment_groups = {}
    for item in equipment:
        # Simple categorization based on keywords
        if any(keyword in item.lower() for keyword in ["safety", "ppe", "protective"]):
            category = "Safety Equipment"
        elif any(keyword in item.lower() for keyword in ["tool", "wrench", "screwdriver", "hammer"]):
            category = "Tools"
        elif any(keyword in item.lower() for keyword in ["machine", "equipment", "device"]):
            category = "Machinery"
        elif any(keyword in item.lower() for keyword in ["material", "chemical", "substance"]):
            category = "Materials"
        else:
            category = "General"
        
        if category not in equipment_groups:
            equipment_groups[category] = []
        equipment_groups[category].append(item)
    
    # Display grouped equipment
    for category, items in equipment_groups.items():
        with st.expander(f"{category} ({len(items)} items)"):
            for item in items:
                st.write(f"• {item}")


def render_metadata(sop_data: Dict[str, Any]):
    """Render SOP metadata and additional information"""
    st.markdown("### 📊 Document Metadata")
    
    # Source information
    source_info = sop_data.get("source", {})
    if source_info:
        st.markdown("**Source Information:**")
        st.json(source_info)
    
    # Definitions and glossary
    definitions = sop_data.get("definitions_glossary", [])
    if definitions:
        st.markdown("**Definitions & Glossary:**")
        for definition in definitions:
            term = definition.get("term", "Unknown")
            meaning = definition.get("definition", "No definition")
            st.write(f"**{term}:** {meaning}")
    
    # Preconditions
    preconditions = sop_data.get("preconditions", [])
    if preconditions:
        st.markdown("**Preconditions:**")
        for condition in preconditions:
            st.write(f"- {condition}")
    
    # Acceptance criteria
    acceptance_criteria = sop_data.get("acceptance_criteria", [])
    if acceptance_criteria:
        st.markdown("**Acceptance Criteria:**")
        for criteria in acceptance_criteria:
            st.write(f"- {criteria}")
    
    # Compliance references
    compliance_refs = sop_data.get("compliance_refs", [])
    if compliance_refs:
        st.markdown("**Compliance References:**")
        for ref in compliance_refs:
            st.write(f"- {ref}")
    
    # Change log
    change_log = sop_data.get("change_log", [])
    if change_log:
        st.markdown("**Change Log:**")
        for change in change_log:
            date = change.get("date", "Unknown")
            version = change.get("version", "Unknown")
            description = change.get("description", "No description")
            st.write(f"**{version}** ({date}): {description}")


def render_sop_comparison(sop1: Dict[str, Any], sop2: Dict[str, Any]):
    """
    Render a comparison between two SOP documents.
    
    Requirements: 4.2 - version comparison functionality
    """
    st.markdown("### 📊 SOP Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Document A:** {sop1.get('title', 'Unknown')}")
        st.write(f"Revision: {sop1.get('revision', 'N/A')}")
        st.write(f"Effective Date: {sop1.get('effective_date', 'N/A')}")
    
    with col2:
        st.markdown(f"**Document B:** {sop2.get('title', 'Unknown')}")
        st.write(f"Revision: {sop2.get('revision', 'N/A')}")
        st.write(f"Effective Date: {sop2.get('effective_date', 'N/A')}")
    
    # Compare key sections
    comparison_sections = [
        ("procedure_steps", "Procedure Steps"),
        ("risks", "Risks"),
        ("controls", "Controls"),
        ("roles_responsibilities", "Roles & Responsibilities"),
        ("materials_equipment", "Equipment & Materials")
    ]
    
    for section_key, section_name in comparison_sections:
        st.markdown(f"#### {section_name}")
        
        data1 = sop1.get(section_key, [])
        data2 = sop2.get(section_key, [])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Document A:**")
            if data1:
                for item in data1[:5]:  # Show first 5 items
                    if isinstance(item, dict):
                        display_text = item.get("description", str(item))
                    else:
                        display_text = str(item)
                    st.write(f"- {display_text[:100]}...")
                if len(data1) > 5:
                    st.caption(f"... and {len(data1) - 5} more items")
            else:
                st.info("No data")
        
        with col2:
            st.markdown("**Document B:**")
            if data2:
                for item in data2[:5]:  # Show first 5 items
                    if isinstance(item, dict):
                        display_text = item.get("description", str(item))
                    else:
                        display_text = str(item)
                    st.write(f"- {display_text[:100]}...")
                if len(data2) > 5:
                    st.caption(f"... and {len(data2) - 5} more items")
            else:
                st.info("No data")
        
        # Highlight differences
        if len(data1) != len(data2):
            if len(data1) > len(data2):
                st.warning(f"Document A has {len(data1) - len(data2)} more {section_name.lower()}")
            else:
                st.warning(f"Document B has {len(data2) - len(data1)} more {section_name.lower()}")
        
        st.divider()


def render_sop_summary_card(sop_data: Dict[str, Any]):
    """
    Render a compact summary card for SOP data.
    
    Requirements: 5.2 - structured SOP card display
    """
    title = sop_data.get("title", "Unknown SOP")
    doc_id = sop_data.get("doc_id", "N/A")
    process_name = sop_data.get("process_name", "N/A")
    
    with st.container():
        st.markdown(f"### 📋 {title}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Document ID", doc_id)
            st.metric("Process", process_name)
        
        with col2:
            step_count = len(sop_data.get("procedure_steps", []))
            risk_count = len(sop_data.get("risks", []))
            st.metric("Steps", step_count)
            st.metric("Risks", risk_count)
        
        with col3:
            control_count = len(sop_data.get("controls", []))
            role_count = len(sop_data.get("roles_responsibilities", []))
            st.metric("Controls", control_count)
            st.metric("Roles", role_count)
        
        # Quick info
        revision = sop_data.get("revision", "N/A")
        effective_date = sop_data.get("effective_date", "N/A")
        owner_role = sop_data.get("owner_role", "N/A")
        
        st.caption(f"Revision: {revision} | Effective: {effective_date} | Owner: {owner_role}")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("View Details", key=f"details_{doc_id}"):
                st.session_state[f"show_details_{doc_id}"] = True
        
        with col2:
            if st.button("Export", key=f"export_{doc_id}"):
                st.session_state[f"export_{doc_id}"] = True
        
        with col3:
            if st.button("Compare", key=f"compare_{doc_id}"):
                st.session_state[f"compare_{doc_id}"] = True
        
        # Show details if requested
        if st.session_state.get(f"show_details_{doc_id}", False):
            render_sop_card(sop_data, expanded=True)
            if st.button("Hide Details", key=f"hide_{doc_id}"):
                st.session_state[f"show_details_{doc_id}"] = False