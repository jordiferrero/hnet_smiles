"""
Utility functions for visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Optional


def bytes_to_hex(text: str) -> str:
    """Convert text to hexadecimal representation."""
    return ' '.join([f'{b:02X}' for b in text.encode('utf-8')])


def get_chunk_spans(boundary_mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Get chunk spans from boundary mask.
    
    Args:
        boundary_mask: Boolean array where True indicates chunk boundaries
    
    Returns:
        List of (start, end) tuples for each chunk
    """
    spans = []
    start = 0
    
    for i, is_boundary in enumerate(boundary_mask):
        if is_boundary and i > start:
            spans.append((start, i))
            start = i
    
    # Add final chunk
    if start < len(boundary_mask):
        spans.append((start, len(boundary_mask)))
    
    return spans


def create_chunking_colormap():
    """Create a colormap for chunking visualization."""
    colors = ['#FFFFFF', '#90EE90', '#4169E1']  # White, Light Green, Blue
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('chunking', colors, N=n_bins)
    return cmap


def setup_figure(
    text: str,
    hex_encoding: str,
    num_chars: int,
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 100,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Setup figure for chunking visualization.
    
    Returns:
        Figure and list of axes
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, dpi=dpi)
    
    # Top: Hex encoding
    axes[0].axis('off')
    axes[0].text(0.5, 0.5, hex_encoding, 
                ha='center', va='center', 
                fontfamily='monospace', fontsize=8,
                transform=axes[0].transAxes)
    axes[0].set_title('Hex Encoding', fontsize=10, pad=5)
    
    # Middle: Text with boundaries
    axes[1].axis('off')
    axes[1].text(0.5, 0.5, text[:num_chars] if num_chars < len(text) else text,
                ha='center', va='center',
                fontfamily='monospace', fontsize=12,
                transform=axes[1].transAxes)
    axes[1].set_title('SMILES String', fontsize=10, pad=5)
    
    # Bottom: Chunking visualization
    axes[2].set_xlim(0, len(text))
    axes[2].set_ylim(-0.5, 1.5)
    axes[2].set_aspect('equal')
    axes[2].axis('off')
    axes[2].set_title('Dynamic Chunking', fontsize=10, pad=5)
    
    plt.tight_layout()
    
    return fig, axes


def draw_chunking_visualization(
    axes: plt.Axes,
    text: str,
    boundary_mask: np.ndarray,
    boundary_prob: Optional[np.ndarray] = None,
    current_length: Optional[int] = None,
    square_size: float = 0.8,
    stage_idx: int = 0,
):
    """
    Draw chunking visualization on axes.
    
    Args:
        axes: Matplotlib axes to draw on
        text: Input text
        boundary_mask: Boolean array marking chunk boundaries
        boundary_prob: Optional probability array
        current_length: Current length to visualize (for progressive display)
        square_size: Size of squares
        stage_idx: Stage index for color differentiation (0=blue, 1=purple)
    """
    if current_length is None:
        current_length = len(text)
    
    num_chars = min(current_length, len(text))
    boundary_mask = boundary_mask[:num_chars]
    
    # Stage-specific colors
    stage_colors = {
        0: {'boundary': '#90EE90', 'span_line': 'b', 'span_fill': '#ADD8E6'},  # Green/Blue for stage 0
        1: {'boundary': '#DDA0DD', 'span_line': 'purple', 'span_fill': '#E6E6FA'},  # Plum/Purple for stage 1
    }
    colors = stage_colors.get(stage_idx, stage_colors[0])
    
    # Draw top row: boundary markers (colored squares)
    y_top = 1.0
    for i in range(num_chars):
        color = colors['boundary'] if boundary_mask[i] else '#FFFFFF'
        square = mpatches.Rectangle(
            (i, y_top - square_size/2),
            square_size, square_size,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5
        )
        axes.add_patch(square)
    
    # Draw bottom row: chunk spans (U-shapes)
    y_bottom = 0.0
    spans = get_chunk_spans(boundary_mask)
    
    for start, end in spans:
        if end > num_chars:
            end = num_chars
        
        # Draw U-shape
        width = end - start
        if width > 0:
            # Left vertical line
            axes.plot([start, start], [y_bottom, y_bottom + square_size], 
                     color=colors['span_line'], linestyle='-', linewidth=2, alpha=0.7)
            # Bottom horizontal line
            axes.plot([start, end], [y_bottom, y_bottom], 
                     color=colors['span_line'], linestyle='-', linewidth=2, alpha=0.7)
            # Right vertical line
            axes.plot([end, end], [y_bottom, y_bottom + square_size], 
                     color=colors['span_line'], linestyle='-', linewidth=2, alpha=0.7)
            
            # Fill with light color
            rect = mpatches.Rectangle(
                (start, y_bottom),
                width, square_size,
                facecolor=colors['span_fill'],
                alpha=0.3,
                edgecolor='none'
            )
            axes.add_patch(rect)
    
    # Draw white squares in bottom row
    for i in range(num_chars):
        square = mpatches.Rectangle(
            (i, y_bottom),
            square_size, square_size,
            facecolor='#FFFFFF',
            edgecolor='black',
            linewidth=0.5
        )
        axes.add_patch(square)
    
    # Add character positions
    for i in range(min(num_chars, len(text))):
        axes.text(i + 0.4, y_bottom + square_size/2, text[i],
                 ha='center', va='center', fontsize=8)


def draw_multistage_chunking_visualization(
    axes: plt.Axes,
    text: str,
    stages_data: List[dict],
    current_length: Optional[int] = None,
    square_size: float = 0.8,
):
    """
    Draw multi-stage chunking visualization on axes (for 2-stage H-Net).
    
    NOTE: In 2-stage H-Net, stages operate at DIFFERENT granularities:
    - Stage 0: Operates on original byte sequence (length = text length)
    - Stage 1: Operates on Stage 0's chunks (length = number of Stage 0 chunks)
    
    This function visualizes Stage 0 at character level, and maps Stage 1 boundaries
    back to character positions by overlaying them on Stage 0 chunks.
    
    Args:
        axes: Matplotlib axes to draw on
        text: Input text
        stages_data: List of dicts with 'boundary_mask' and 'boundary_prob' for each stage
        current_length: Current length to visualize (for progressive display)
        square_size: Size of squares
    """
    if current_length is None:
        current_length = len(text)
    
    num_chars = min(current_length, len(text))
    num_stages = len(stages_data)
    
    # Stage-specific colors
    stage_colors = [
        {'boundary': '#90EE90', 'span_line': 'b', 'span_fill': '#ADD8E6', 'label': 'Stage 0 (Bytes→Chunks)'},
        {'boundary': '#DDA0DD', 'span_line': 'purple', 'span_fill': '#E6E6FA', 'label': 'Stage 1 (Chunks→SuperChunks)'},
    ]
    
    # Get Stage 0 data (always at character level)
    stage0_mask = stages_data[0]['boundary_mask']
    stage0_mask_clipped = stage0_mask[:num_chars] if len(stage0_mask) >= num_chars else stage0_mask
    stage0_spans = get_chunk_spans(stage0_mask_clipped)
    stage0_num_chunks = len(stage0_spans)
    
    # Calculate vertical layout
    y_stage0_boundary = 2.5
    y_stage0_spans = 1.5
    y_stage1_indicator = 0.5 if num_stages > 1 else None
    y_chars = -0.5
    
    colors0 = stage_colors[0]
    
    # ===== Draw Stage 0: Byte-level chunking =====
    # Draw Stage 0 boundary markers
    for i in range(len(stage0_mask_clipped)):
        color = colors0['boundary'] if stage0_mask_clipped[i] else '#FFFFFF'
        square = mpatches.Rectangle(
            (i, y_stage0_boundary - square_size/2),
            square_size, square_size,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5
        )
        axes.add_patch(square)
    
    # Draw Stage 0 chunk spans
    for start, end in stage0_spans:
        if end > num_chars:
            end = num_chars
        width = end - start
        if width > 0:
            axes.plot([start, start], [y_stage0_spans, y_stage0_spans + square_size], 
                     color=colors0['span_line'], linestyle='-', linewidth=2, alpha=0.7)
            axes.plot([start, end], [y_stage0_spans, y_stage0_spans], 
                     color=colors0['span_line'], linestyle='-', linewidth=2, alpha=0.7)
            axes.plot([end, end], [y_stage0_spans, y_stage0_spans + square_size], 
                     color=colors0['span_line'], linestyle='-', linewidth=2, alpha=0.7)
            rect = mpatches.Rectangle(
                (start, y_stage0_spans), width, square_size,
                facecolor=colors0['span_fill'], alpha=0.3, edgecolor='none'
            )
            axes.add_patch(rect)
    
    # Add Stage 0 label
    axes.text(-1.5, y_stage0_boundary, colors0['label'], 
             ha='right', va='center', fontsize=7, fontweight='bold',
             color=colors0['span_line'])
    
    # ===== Draw Stage 1: Map chunk-level boundaries back to character positions =====
    if num_stages > 1 and y_stage1_indicator is not None:
        colors1 = stage_colors[1]
        stage1_mask = stages_data[1]['boundary_mask']
        
        # Stage 1 boundary mask is over the CHUNKS from Stage 0, not characters
        # Map Stage 1 boundaries back to character positions
        # Each Stage 1 boundary at position i corresponds to Stage 0's chunk i
        
        # Draw Stage 1 super-chunk spans by grouping Stage 0 chunks
        if len(stage1_mask) > 0 and len(stage0_spans) > 0:
            # Get Stage 1 spans (over chunk indices)
            stage1_spans = get_chunk_spans(stage1_mask)
            
            # Map each Stage 1 span to character positions using Stage 0 spans
            for chunk_start, chunk_end in stage1_spans:
                # chunk_start and chunk_end are indices into Stage 0's chunks
                if chunk_start < len(stage0_spans) and chunk_end <= len(stage0_spans):
                    # Get character positions from Stage 0 spans
                    char_start = stage0_spans[chunk_start][0]
                    char_end = stage0_spans[min(chunk_end, len(stage0_spans)) - 1][1] if chunk_end > 0 else char_start
                    
                    if char_end > num_chars:
                        char_end = num_chars
                    
                    width = char_end - char_start
                    if width > 0:
                        # Draw thicker purple bracket for super-chunks
                        axes.plot([char_start, char_start], [y_stage1_indicator, y_stage1_indicator + square_size], 
                                 color=colors1['span_line'], linestyle='-', linewidth=3, alpha=0.8)
                        axes.plot([char_start, char_end], [y_stage1_indicator, y_stage1_indicator], 
                                 color=colors1['span_line'], linestyle='-', linewidth=3, alpha=0.8)
                        axes.plot([char_end, char_end], [y_stage1_indicator, y_stage1_indicator + square_size], 
                                 color=colors1['span_line'], linestyle='-', linewidth=3, alpha=0.8)
                        
                        rect = mpatches.Rectangle(
                            (char_start, y_stage1_indicator), width, square_size,
                            facecolor=colors1['span_fill'], alpha=0.4, edgecolor='none'
                        )
                        axes.add_patch(rect)
        
        # Add Stage 1 label with chunk count info
        stage1_info = f"{colors1['label']} ({len(stage1_mask)} chunks → {sum(stage1_mask)} super-chunks)"
        axes.text(-1.5, y_stage1_indicator + square_size/2, stage1_info, 
                 ha='right', va='center', fontsize=6, fontweight='bold',
                 color=colors1['span_line'])
    
    # ===== Draw character row at the bottom =====
    for i in range(num_chars):
        square = mpatches.Rectangle(
            (i, y_chars),
            square_size, square_size,
            facecolor='#FFFFFF',
            edgecolor='black',
            linewidth=0.5
        )
        axes.add_patch(square)
        axes.text(i + 0.4, y_chars + square_size/2, text[i],
                 ha='center', va='center', fontsize=8)


def create_animation_frame(
    text: str,
    hex_encoding: str,
    boundary_mask: np.ndarray,
    boundary_prob: Optional[np.ndarray] = None,
    current_length: Optional[int] = None,
    frame_num: int = 0,
    stages_data: Optional[List[dict]] = None,
) -> plt.Figure:
    """
    Create a single animation frame.
    Supports both single-stage (backward compat) and multi-stage visualization.
    
    Args:
        text: Input text
        hex_encoding: Hex encoding of text
        boundary_mask: Boundary mask for stage 0 (for backward compat)
        boundary_prob: Boundary probabilities for stage 0 (for backward compat)
        current_length: Current length to visualize
        frame_num: Frame number
        stages_data: Optional list of stage dicts for multi-stage visualization
    
    Returns:
        Matplotlib figure
    """
    # Determine number of stages
    num_stages = len(stages_data) if stages_data else 1
    
    # Adjust figure height for multi-stage
    if num_stages > 1:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), dpi=100, 
                                  gridspec_kw={'height_ratios': [1, 1, num_stages * 1.5]})
    else:
        fig, axes = setup_figure(text, hex_encoding, len(text))
    
    # Update text display
    display_length = current_length if current_length else len(text)
    axes[1].clear()
    axes[1].axis('off')
    axes[1].text(0.5, 0.5, text[:display_length],
                ha='center', va='center',
                fontfamily='monospace', fontsize=12,
                transform=axes[1].transAxes)
    
    # Draw chunking visualization
    axes[2].clear()
    axes[2].set_xlim(-8, len(text) + 1)  # Extra space for labels on left
    
    if stages_data and num_stages > 1:
        # Multi-stage visualization
        # Layout: Stage0 boundary (2.5), Stage0 spans (1.5), Stage1 spans (0.5), chars (-0.5)
        axes[2].set_ylim(-1.5, 4.0)
        axes[2].set_aspect('equal')
        axes[2].axis('off')
        
        draw_multistage_chunking_visualization(
            axes[2], text, stages_data, current_length
        )
        stage_str = f"{num_stages} stages"
    else:
        # Single-stage visualization (backward compat)
        axes[2].set_ylim(-0.5, 1.5)
        axes[2].set_aspect('equal')
        axes[2].axis('off')
        
        draw_chunking_visualization(
            axes[2], text, boundary_mask, boundary_prob, current_length
        )
        stage_str = "1 stage"
    
    # Add frame info
    fig.suptitle(f'Frame {frame_num}: Processing {display_length}/{len(text)} characters ({stage_str})',
                 fontsize=12, y=0.98)
    
    plt.tight_layout()
    
    return fig

