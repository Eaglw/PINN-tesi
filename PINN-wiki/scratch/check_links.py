import os
import re

wiki_dir = r"c:\Users\eaglw\Documents\PINN tesi\PINN-wiki\Wiki"
all_links = set()
all_pages = set()

# 1. Collect all existing pages (basenames without extension)
for root, dirs, files in os.walk(wiki_dir):
    for file in files:
        if file.endswith(".md"):
            all_pages.add(file[:-3])

# 2. Collect all wikilinks within pages
link_pattern = re.compile(r"\[\[([^\]]+)\]\]")

broken_links = []

for root, dirs, files in os.walk(wiki_dir):
    for file in files:
        if file.endswith(".md"):
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                content = f.read()
                links = link_pattern.findall(content)
                for link in links:
                    # Handle aliases [[Link|Alias]]
                    link_target = link.split('|')[0]
                    # Handle anchors [[Link#Anchor]]
                    link_target = link_target.split('#')[0]
                    
                    # Ignore empty targets (just an anchor within same page [[#Anchor]])
                    if not link_target:
                        continue
                        
                    # Ignore .pdf links as they point to Reference/ (standard allowed per GEMINI.md)
                    if link_target.lower().endswith(".pdf"):
                        continue
                    if link_target not in all_pages:
                        broken_links.append((file, link_target))

if broken_links:
    print("Found broken links:")
    for source, target in sorted(set(broken_links)):
        print(f"- In {source}: [[{target}]]")
else:
    print("No broken links found!")
