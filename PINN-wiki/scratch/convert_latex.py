import os
import re

wiki_root = r"c:\Users\eaglw\Documents\PINN tesi\PINN-wiki\Wiki"

def convert_latex_to_obsidian(content):
    # Convert blocks: \[ ... \] to $$ ... $$
    content = re.sub(r'\\\[', '$$', content)
    content = re.sub(r'\\\]', '$$', content)
    # Convert inline: \( ... \) to $ ... $
    content = re.sub(r'\\\(', '$', content)
    content = re.sub(r'\\\)', '$', content)
    return content

for root, dirs, filenames in os.walk(wiki_root):
    for f in filenames:
        if f.endswith(".md"):
            path = os.path.join(root, f)
            with open(path, 'r', encoding='utf-8') as file:
                original = file.read()
            
            converted = convert_latex_to_obsidian(original)
            
            if converted != original:
                with open(path, 'w', encoding='utf-8') as file:
                    file.write(converted)
                print(f"Converted: {f}")

# Also update GEMINI.md in the root of PINN-wiki
gemini_wiki_path = r"c:\Users\eaglw\Documents\PINN tesi\PINN-wiki\GEMINI.md"
if os.path.exists(gemini_wiki_path):
    with open(gemini_wiki_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Update the formatting rule
    content = content.replace(r"\( ... \)", "$ ... $")
    content = content.replace(r"\[ ... \]", "$$ ... $$")
    
    with open(gemini_wiki_path, 'w', encoding='utf-8') as file:
        file.write(content)
    print("Updated GEMINI.md")
