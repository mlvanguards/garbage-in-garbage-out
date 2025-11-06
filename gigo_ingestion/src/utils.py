import base64
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def encode_image_to_data_uri(image_path: Path) -> str:
    """
    Read a PNG (or other) image from disk and return a data URI string.
    Raises FileNotFoundError if the file does not exist.
    """
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{base64_image}"


def load_image_as_data_uri(path: Optional[Path], required: bool = True) -> str:
    """
    Load an image and encode it as a data URI.
    
    Args:
        path: Path to the image file
        required: If True, raise exception on missing file; if False, return empty string
        
    Returns:
        Base64-encoded data URI, or empty string if not found and not required
        
    Raises:
        FileNotFoundError: If required=True and file doesn't exist
    """
    if path is None:
        if required:
            raise FileNotFoundError("Image path is required but was None")
        return ""
    
    try:
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Image file not found: {path}")
            return ""
        return encode_image_to_data_uri(path)
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.debug(f"Could not load image {path}: {e}")
        if required:
            raise
        return ""


def load_json_file(path: Optional[Path], required: bool = True) -> str:
    """
    Load a JSON file and return as string.
    
    Args:
        path: Path to the JSON file
        required: If True, raise exception on missing file; if False, return "{}"
        
    Returns:
        JSON content as string, or "{}" if not found and not required
        
    Raises:
        FileNotFoundError: If required=True and file doesn't exist
    """
    if path is None:
        if required:
            raise FileNotFoundError("JSON path is required but was None")
        return "{}"
    
    try:
        if not path.exists():
            if required:
                raise FileNotFoundError(f"JSON file not found: {path}")
            return "{}"
        
        with open(path, "r") as f:
            data = json.load(f)
            return json.dumps(data)
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.debug(f"Could not load JSON {path}: {e}")
        if required:
            raise
        return "{}"


def load_text_file(path: Optional[Path], required: bool = True) -> str:
    """
    Load a text file and return as string.
    
    Args:
        path: Path to the text file
        required: If True, raise exception on missing file; if False, return ""
        
    Returns:
        Text content as string, or "" if not found and not required
        
    Raises:
        FileNotFoundError: If required=True and file doesn't exist
    """
    if path is None:
        if required:
            raise FileNotFoundError("Text path is required but was None")
        return ""
    
    try:
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Text file not found: {path}")
            return ""
        
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.debug(f"Could not load text {path}: {e}")
        if required:
            raise
        return ""


if __name__ == "__main__":
    print(encode_image_to_data_uri(Path("tests/data/images/table_1.png")))