"""
UI Components and Utilities

Reusable UI components and utility functions for the Streamlit frontend.

Requirements: 5.1, 5.3, 5.4
"""

import streamlit as st
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def show_loading_spinner(message: str = "Loading..."):
    """Show a loading spinner with custom message"""
    return st.spinner(message)


def show_progress_bar(progress: float, message: str = ""):
    """Show a progress bar with optional message"""
    progress_bar = st.progress(progress)
    if message:
        st.text(message)
    return progress_bar


def show_status_indicator(status: str, message: str = ""):
    """Show a status indicator with color coding"""
    status_colors = {
        "success": "🟢",
        "warning": "🟡", 
        "error": "🔴",
        "info": "🔵",
        "running": "🟡"
    }
    
    icon = status_colors.get(status.lower(), "⚪")
    st.markdown(f"{icon} **{status.title()}** {message}")


def create_metric_card(title: str, value: str, delta: Optional[str] = None, help_text: Optional[str] = None):
    """Create a metric card with optional delta and help text"""
    st.metric(
        label=title,
        value=value,
        delta=delta,
        help=help_text
    )


def create_info_card(title: str, content: str, card_type: str = "info"):
    """Create an information card with different styling"""
    card_functions = {
        "info": st.info,
        "success": st.success,
        "warning": st.warning,
        "error": st.error
    }
    
    card_func = card_functions.get(card_type, st.info)
    card_func(f"**{title}**\n\n{content}")


def create_expandable_section(title: str, content_func: Callable, expanded: bool = False, key: Optional[str] = None):
    """Create an expandable section with custom content"""
    with st.expander(title, expanded=expanded):
        content_func()


def create_tabs_section(tab_configs: List[Dict[str, Any]]):
    """
    Create a tabs section with multiple tabs.
    
    Args:
        tab_configs: List of dicts with 'title' and 'content_func' keys
    """
    tab_titles = [config["title"] for config in tab_configs]
    tabs = st.tabs(tab_titles)
    
    for tab, config in zip(tabs, tab_configs):
        with tab:
            config["content_func"]()


def create_filter_sidebar(filters_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a standardized filter sidebar.
    
    Args:
        filters_config: Configuration for filters
        
    Returns:
        Dict of selected filter values
    """
    st.sidebar.header("🔍 Filters")
    
    selected_filters = {}
    
    for filter_name, filter_config in filters_config.items():
        filter_type = filter_config.get("type", "selectbox")
        label = filter_config.get("label", filter_name.title())
        options = filter_config.get("options", [])
        default = filter_config.get("default")
        help_text = filter_config.get("help")
        
        if filter_type == "selectbox":
            selected_filters[filter_name] = st.sidebar.selectbox(
                label, options, index=options.index(default) if default in options else 0, help=help_text
            )
        elif filter_type == "multiselect":
            selected_filters[filter_name] = st.sidebar.multiselect(
                label, options, default=default, help=help_text
            )
        elif filter_type == "text_input":
            selected_filters[filter_name] = st.sidebar.text_input(
                label, value=default or "", help=help_text
            )
        elif filter_type == "slider":
            min_val = filter_config.get("min", 0)
            max_val = filter_config.get("max", 100)
            selected_filters[filter_name] = st.sidebar.slider(
                label, min_val, max_val, default or min_val, help=help_text
            )
        elif filter_type == "checkbox":
            selected_filters[filter_name] = st.sidebar.checkbox(
                label, value=default or False, help=help_text
            )
    
    return selected_filters


def create_data_table(data: List[Dict[str, Any]], columns: Optional[List[str]] = None, 
                     sortable: bool = True, filterable: bool = True):
    """Create a data table with optional sorting and filtering"""
    if not data:
        st.info("No data to display")
        return
    
    df = pd.DataFrame(data)
    
    if columns:
        df = df[columns]
    
    if filterable:
        # Add simple text filter
        filter_text = st.text_input("Filter table:", placeholder="Type to filter...")
        if filter_text:
            # Simple text-based filtering across all columns
            mask = df.astype(str).apply(lambda x: x.str.contains(filter_text, case=False, na=False)).any(axis=1)
            df = df[mask]
    
    st.dataframe(df, use_container_width=True)
    
    return df


def create_confidence_chart(confidence_scores: List[float], labels: Optional[List[str]] = None):
    """Create a confidence score visualization"""
    if not confidence_scores:
        st.info("No confidence data to display")
        return
    
    # Create confidence level categories
    confidence_levels = []
    colors = []
    
    for score in confidence_scores:
        if score >= 0.7:
            confidence_levels.append("High")
            colors.append("green")
        elif score >= 0.4:
            confidence_levels.append("Medium") 
            colors.append("orange")
        else:
            confidence_levels.append("Low")
            colors.append("red")
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=labels or list(range(len(confidence_scores))),
            y=confidence_scores,
            marker_color=colors,
            text=[f"{score:.2f}" for score in confidence_scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Confidence Scores",
        xaxis_title="Questions" if not labels else "Items",
        yaxis_title="Confidence Score",
        yaxis=dict(range=[0, 1]),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_response_time_chart(response_times: List[int], labels: Optional[List[str]] = None):
    """Create a response time visualization"""
    if not response_times:
        st.info("No response time data to display")
        return
    
    fig = go.Figure(data=[
        go.Scatter(
            x=labels or list(range(len(response_times))),
            y=response_times,
            mode='lines+markers',
            name='Response Time',
            line=dict(color='blue'),
            marker=dict(size=8)
        )
    ])
    
    fig.update_layout(
        title="Response Times",
        xaxis_title="Questions" if not labels else "Items",
        yaxis_title="Response Time (ms)",
        showlegend=False
    )
    
    # Add average line
    avg_time = sum(response_times) / len(response_times)
    fig.add_hline(y=avg_time, line_dash="dash", line_color="red", 
                  annotation_text=f"Average: {avg_time:.0f}ms")
    
    st.plotly_chart(fig, use_container_width=True)


def create_citation_network(citations: List[Dict[str, Any]]):
    """Create a network visualization of document citations"""
    if not citations:
        st.info("No citation data to display")
        return
    
    # Count citations per document
    doc_counts = {}
    for citation in citations:
        doc_id = citation.get("doc_id", "Unknown")
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
    
    # Create pie chart
    fig = px.pie(
        values=list(doc_counts.values()),
        names=list(doc_counts.keys()),
        title="Citations by Document"
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_export_buttons(data: Any, filename_prefix: str = "export"):
    """Create standardized export buttons for different formats"""
    if not data:
        st.info("No data to export")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export CSV"):
            if isinstance(data, list) and data:
                df = pd.DataFrame(data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with col2:
        if st.button("📄 Export JSON"):
            import json
            json_str = json.dumps(data, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("📝 Export Text"):
            if isinstance(data, list):
                text_content = "\n".join([str(item) for item in data])
            else:
                text_content = str(data)
            
            st.download_button(
                label="Download Text",
                data=text_content,
                file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )


def create_search_interface(placeholder: str = "Search...", help_text: Optional[str] = None):
    """Create a standardized search interface"""
    col1, col2 = st.columns([4, 1])
    
    with col1:
        search_query = st.text_input(
            "Search",
            placeholder=placeholder,
            help=help_text,
            label_visibility="collapsed"
        )
    
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    return search_query, search_button


def create_pagination(total_items: int, items_per_page: int = 10, key: str = "pagination"):
    """Create pagination controls"""
    if total_items <= items_per_page:
        return 0, items_per_page
    
    total_pages = (total_items - 1) // items_per_page + 1
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("◀ Previous", key=f"{key}_prev", disabled=st.session_state.get(f"{key}_page", 0) == 0):
            st.session_state[f"{key}_page"] = max(0, st.session_state.get(f"{key}_page", 0) - 1)
    
    with col2:
        current_page = st.session_state.get(f"{key}_page", 0)
        st.write(f"Page {current_page + 1} of {total_pages}")
    
    with col3:
        if st.button("Next ▶", key=f"{key}_next", disabled=current_page >= total_pages - 1):
            st.session_state[f"{key}_page"] = min(total_pages - 1, current_page + 1)
    
    start_idx = current_page * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    
    return start_idx, end_idx


def create_toast_notification(message: str, notification_type: str = "info", duration: int = 3):
    """Create a toast notification (using Streamlit's built-in notifications)"""
    if notification_type == "success":
        st.success(message)
    elif notification_type == "warning":
        st.warning(message)
    elif notification_type == "error":
        st.error(message)
    else:
        st.info(message)
    
    # Auto-clear after duration (simplified approach)
    if duration > 0:
        time.sleep(duration)


def create_confirmation_dialog(message: str, key: str) -> bool:
    """Create a confirmation dialog"""
    if st.button(f"⚠️ {message}", key=f"{key}_trigger"):
        st.session_state[f"{key}_confirm"] = True
    
    if st.session_state.get(f"{key}_confirm", False):
        st.warning(f"Are you sure? {message}")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Yes", key=f"{key}_yes"):
                st.session_state[f"{key}_confirm"] = False
                return True
        
        with col2:
            if st.button("❌ No", key=f"{key}_no"):
                st.session_state[f"{key}_confirm"] = False
    
    return False


def create_file_uploader_with_preview(
    label: str,
    accepted_types: List[str],
    max_size_mb: int = 50,
    multiple: bool = True
):
    """Create a file uploader with preview capabilities"""
    uploaded_files = st.file_uploader(
        label,
        type=accepted_types,
        accept_multiple_files=multiple,
        help=f"Maximum file size: {max_size_mb}MB"
    )
    
    if uploaded_files:
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]
        
        st.write("**Uploaded Files:**")
        for file in uploaded_files:
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write(f"📄 {file.name}")
            
            with col2:
                file_size_mb = len(file.getvalue()) / (1024 * 1024)
                st.write(f"{file_size_mb:.2f} MB")
            
            with col3:
                if file_size_mb > max_size_mb:
                    st.error("Too large!")
                else:
                    st.success("✅ Valid")
    
    return uploaded_files


def create_mode_switcher(current_mode: str, on_change: Optional[Callable] = None):
    """Create a mode switcher component"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        aws_selected = st.button(
            "☁️ AWS Mode",
            use_container_width=True,
            type="primary" if current_mode == "aws" else "secondary"
        )
    
    with col2:
        local_selected = st.button(
            "💻 Local Mode", 
            use_container_width=True,
            type="primary" if current_mode == "local" else "secondary"
        )
    
    if aws_selected and current_mode != "aws":
        if on_change:
            on_change("aws")
        return "aws"
    elif local_selected and current_mode != "local":
        if on_change:
            on_change("local")
        return "local"
    
    return current_mode