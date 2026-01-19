import re
import os
import sys


def convert_headings(input_file_path, output_file_path=None):
    """
    convert_headings converts markdown headings from format === and ---
    to the standard format # and ## respectively.
    """
    pattern_headline_1 = re.compile(r"^(.*)\n(\=+)$", re.MULTILINE)
    pattern_headline_2 = re.compile(r"^(.*)\n(\-+)$", re.MULTILINE)

    with open(input_file_path, "r", encoding="utf-8") as file:
        content = file.read()

    content = pattern_headline_1.sub(r"# \1", content)
    content = pattern_headline_2.sub(r"## \1", content)

    with open(output_file_path or input_file_path, "w", encoding="utf-8") as file:
        file.write(content)


def walk_dir(dir_path):
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".md"):
                convert_headings(os.path.join(root, file))
                

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 convert_headings.py <file or directory>...")
        sys.exit(1)
    for arg in sys.argv[1:]:
        if not os.path.exists(arg):
            print(f"File or directory {arg} does not exist.")
            continue
        if os.path.isdir(arg):
            walk_dir(arg)
        else:
            convert_headings(arg)