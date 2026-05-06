import os
import re

wiki_root = r"c:\Users\eaglw\Documents\PINN tesi\PINN-wiki\Wiki"
index_path = os.path.join(wiki_root, "00_Index.md")

def get_all_wiki_files():
    files = {}
    for root, dirs, filenames in os.walk(wiki_root):
        for f in filenames:
            if f.endswith(".md"):
                rel_path = os.path.relpath(os.path.join(root, f), wiki_root)
                name = f[:-3]
                files[name] = rel_path
    return files

def get_index_links():
    if not os.path.exists(index_path):
        return set()
    with open(index_path, 'r', encoding='utf-8') as f:
        content = f.read()
    links = re.findall(r'\[\[([^]|]+)(?:\|[^]]+)?\]\]', content)
    return set(links)

def check_broken_links(all_files):
    broken_links = {}
    for name, rel_path in all_files.items():
        full_path = os.path.join(wiki_root, rel_path)
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        links = re.findall(r'\[\[([^]|]+)(?:\|[^]]+)?\]\]', content)
        for link in links:
            if link not in all_files and not link.endswith(".pdf"):
                if name not in broken_links:
                    broken_links[name] = []
                broken_links[name].append(link)
    return broken_links

all_files = get_all_wiki_files()
index_links = get_index_links()
broken_links = check_broken_links(all_files)

print("--- WIKI PAGES ---")
for name in sorted(all_files.keys()):
    status = "[INDEXED]" if name in index_links or name in ["00_Index", "01_Log"] else "[MISSING FROM INDEX]"
    print(f"{name:40} {status}")

print("\n--- BROKEN LINKS ---")
if not broken_links:
    print("None found!")
else:
    for page, targets in broken_links.items():
        print(f"In {page}:")
        for t in targets:
            print(f"  -> {t} (Target missing)")

print("\n--- ORPHAN PAGES (Not in Index) ---")
orphans = [f for f in all_files if f not in index_links and f not in ["00_Index", "01_Log"]]
if not orphans:
    print("None found!")
else:
    for o in orphans:
        print(f"  - {o}")
