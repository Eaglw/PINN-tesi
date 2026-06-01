import os
import re
import sys

def check_links():
    wiki_dir = r"c:\Users\eaglw\Documents\PINN tesi\PINN-wiki\Wiki"
    if not os.path.isdir(wiki_dir):
        print(f"Error: {wiki_dir} is not a directory.")
        sys.exit(1)

    # 1. Collect all valid page names (markdown files without extension)
    valid_pages = set()
    for root, dirs, files in os.walk(wiki_dir):
        for file in files:
            if file.endswith(".md"):
                name = file[:-3]
                valid_pages.add(name)

    # 2. Scan each file for [[links]]
    link_pattern = re.compile(r'\[\[([^\]]+)\]\]')
    broken_links = []
    total_links_count = 0

    for root, dirs, files in os.walk(wiki_dir):
        for file in files:
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, wiki_dir)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                links = link_pattern.findall(content)
                for link in links:
                    total_links_count += 1
                    # Handle alias [[Target|Alias]]
                    target = link.split('|')[0].strip()
                    # Handle section anchors [[Target#Section]]
                    target = target.split('#')[0].strip()
                    
                    if not target:
                        continue
                    
                    # Ignore external PDFs or files if they are in specific formats
                    if target.endswith(('.pdf', '.png', '.jpg', '.jpeg')):
                        continue
                        
                    if target not in valid_pages:
                        broken_links.append((rel_path, link, target))

    print(f"Total links checked: {total_links_count}")
    if broken_links:
        print(f"Found {len(broken_links)} broken links:")
        for source, link_text, target in broken_links:
            print(f" - In file '{source}': [[{link_text}]] -> Target '{target}' not found.")
        sys.exit(1)
    else:
        print("Success: All wikilinks are valid! 100% link integrity.")
        sys.exit(0)

if __name__ == "__main__":
    check_links()
