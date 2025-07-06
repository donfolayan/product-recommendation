"""
Project utilities for common operations like finding project root and managing sys.path.
"""
import sys
from pathlib import Path
from typing import Optional


def find_project_root(marker: str = "requirements.txt") -> Path:
    """
    Find the project root directory by looking for a marker file.
    
    Args:
        marker: Name of the marker file to look for (default: requirements.txt)
        
    Returns:
        Path to the project root directory
        
    Raises:
        FileNotFoundError: If project root cannot be found
    """
    p = Path.cwd()
    while not (p / marker).exists() and p != p.parent:
        p = p.parent
    if (p / marker).exists():
        return p
    raise FileNotFoundError(f"Could not find project root with marker '{marker}' from {Path.cwd()}")


def setup_project_path(project_root: Optional[Path] = None) -> Path:
    """
    Set up the project path and add it to sys.path if needed.
    
    Args:
        project_root: Optional project root path. If None, will be auto-detected.
        
    Returns:
        Path to the project root directory
    """
    if project_root is None:
        project_root = find_project_root()
    
    # Add project root to sys.path if not already there
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to the project root directory
    """
    return find_project_root()


def get_relative_path_from_project_root(file_path: Path) -> Path:
    """
    Get the relative path from project root.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Relative path from project root
    """
    project_root = get_project_root()
    return file_path.relative_to(project_root) 