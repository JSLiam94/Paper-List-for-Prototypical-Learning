"""
Script to prefix image paths in the README.md file located in the same directory.

This script searches for Markdown image links of the form:

    ![alt text](image-8.png)

and ensures that the path is prefixed with 'images/', producing:

    ![alt text](images/image-8.png)

It skips links that already start with 'images/'.

Usage:
    python prefix_image_paths.py

This will read and overwrite README.md in the current directory.
"""
import re
from pathlib import Path

README = Path(__file__).parent / "README.md"

def prefix_image_paths(text: str) -> str:
    # Pattern matches ![...](path) where path does not start with images/ or a URL
    pattern = re.compile(r"(!\[[^\]]*\]\()(?!(?:[a-zA-Z]+://|images/))([^\)]+)(\))")
    return pattern.sub(lambda m: f"{m.group(1)}images/{m.group(2)}{m.group(3)}", text)

def main():
    if not README.exists():
        print(f"Error: {README} not found.")
        return
    content = README.read_text(encoding='utf-8')
    updated = prefix_image_paths(content)
    README.write_text(updated, encoding='utf-8')
    print(f"Processed 'README.md', image paths prefixed with 'images/'.")

if __name__ == '__main__':
    main()
