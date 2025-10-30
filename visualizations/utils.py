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
    """
    if current_length is None:
        current_length = len(text)
    
    num_chars = min(current_length, len(text))
    boundary_mask = boundary_mask[:num_chars]
    
    # Draw top row: boundary markers (green squares)
    y_top = 1.0
    for i in range(num_chars):
        color = '#90EE90' if boundary_mask[i] else '#FFFFFF'
        square = mpatches.Rectangle(
            (i, y_top - square_size/2),
            square_size, square_size,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5
        )
        axes.add_patch(square)
    
    # Draw bottom row: chunk spans (blue U-shapes)
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
                     'b-', linewidth=2, alpha=0.7)
            # Bottom horizontal line
            axes.plot([start, end], [y_bottom, y_bottom], 
                     'b-', linewidth=2, alpha=0.7)
            # Right vertical line
            axes.plot([end, end], [y_bottom, y_bottom + square_size], 
                     'b-', linewidth=2, alpha=0.7)
            
            # Fill with light blue
            rect = mpatches.Rectangle(
                (start, y_bottom),
                width, square_size,
                facecolor='#ADD8E6',
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


def create_animation_frame(
    text: str,
    hex_encoding: str,
    boundary_mask: np.ndarray,
    boundary_prob: Optional[np.ndarray] = None,
    current_length: Optional[int] = None,
    frame_num: int = 0,
) -> plt.Figure:
    """
    Create a single animation frame.
    
    Returns:
        Matplotlib figure
    """
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
    axes[2].set_xlim(0, len(text))
    axes[2].set_ylim(-0.5, 1.5)
    axes[2].set_aspect('equal')
    axes[2].axis('off')
    
    draw_chunking_visualization(
        axes[2], text, boundary_mask, boundary_prob, current_length
    )
    
    # Add frame info
    fig.suptitle(f'Frame {frame_num}: Processing {display_length}/{len(text)} characters',
                 fontsize=12, y=0.98)
    
    plt.tight_layout()
    
    return fig

