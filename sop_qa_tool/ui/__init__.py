"""
UI Module for SOP Q&A Tool

Streamlit-based frontend implementation with chat interface, document management,
filtering controls, and export functionality.

Requirements: 5.1, 5.2, 5.3, 5.4, 4.2, 4.3, 7.3
"""

from .streamlit_app import main as run_streamlit_app, APIClient
from .components import (
    create_metric_card,
    create_info_card,
    create_filter_sidebar,
    create_data_table,
    create_confidence_chart,
    create_response_time_chart,
    create_citation_network,
    create_export_buttons,
    create_search_interface,
    create_pagination,
    create_file_uploader_with_preview,
    create_mode_switcher
)
from .sop_cards import (
    render_sop_card,
    render_procedure_steps,
    render_risks_and_controls,
    render_roles_responsibilities,
    render_equipment_materials,
    render_metadata,
    render_sop_comparison,
    render_sop_summary_card
)

__all__ = [
    # Main app
    "run_streamlit_app",
    "APIClient",
    
    # Components
    "create_metric_card",
    "create_info_card", 
    "create_filter_sidebar",
    "create_data_table",
    "create_confidence_chart",
    "create_response_time_chart",
    "create_citation_network",
    "create_export_buttons",
    "create_search_interface",
    "create_pagination",
    "create_file_uploader_with_preview",
    "create_mode_switcher",
    
    # SOP Cards
    "render_sop_card",
    "render_procedure_steps",
    "render_risks_and_controls",
    "render_roles_responsibilities",
    "render_equipment_materials",
    "render_metadata",
    "render_sop_comparison",
    "render_sop_summary_card"
]