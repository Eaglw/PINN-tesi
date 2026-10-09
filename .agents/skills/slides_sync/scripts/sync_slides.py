import os
import sys
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
import re

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

DEFAULT_DOC_ID = "1bTiAcKGMXmr67rRc2oLwJqCIH163FC8jySCspP88fcs"

def sync_presentation(doc_id=DEFAULT_DOC_ID, workspace_root=None):
    if workspace_root is None:
        # Detect root from script location: .agents/skills/slides_sync/scripts/ -> root is 4 levels up
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

    assets_dir = os.path.join(workspace_root, "final_roll", "presentation_assets")
    scratch_dir = os.path.join(workspace_root, "scratch")
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)

    pptx_url = f"https://docs.google.com/presentation/d/{doc_id}/export/pptx"
    pdf_url = f"https://docs.google.com/presentation/d/{doc_id}/export/pdf"

    pptx_path = os.path.join(assets_dir, "Seminario I e II - Tesi PINN.pptx")
    pdf_path = os.path.join(assets_dir, "Seminario I e II - Tesi PINN.pdf")
    old_dump_path = os.path.join(scratch_dir, "slides_dump.txt")
    new_dump_path = os.path.join(scratch_dir, "slides_updated_dump.txt")
    notes_dump_path = os.path.join(scratch_dir, "clean_speaker_notes.txt")

    print(f"[*] Sincronizzazione presentazione Google Slides [ID: {doc_id}]...")
    
    # 1. Download PPTX
    try:
        urllib.request.urlretrieve(pptx_url, pptx_path)
        print(f"[+] PPTX salvato: {pptx_path} ({os.path.getsize(pptx_path):,} bytes)")
    except Exception as e:
        print(f"[!] Errore nel download del PPTX: {e}")
        return False

    # 2. Download PDF
    try:
        urllib.request.urlretrieve(pdf_url, pdf_path)
        print(f"[+] PDF salvato: {pdf_path} ({os.path.getsize(pdf_path):,} bytes)")
    except Exception as e:
        print(f"[!] Warning: Impossibile scaricare PDF ({e}), proseguo con PPTX.")

    # 3. Extract slides and notes from PPTX
    new_slides, new_notes = extract_from_pptx(pptx_path)
    print(f"[+] Estratte {len(new_slides)} slide e {len(new_notes)} note del relatore.")

    # Save new dump
    out_lines = [f"Total slides: {len(new_slides)}\n"]
    for idx in sorted(new_slides.keys()):
        out_lines.append("==============================")
        out_lines.append(f"SLIDE {idx}")
        out_lines.append("==============================")
        out_lines.append(new_slides[idx] + "\n")
    with open(new_dump_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(out_lines))

    # Save notes dump
    notes_lines = []
    for idx in sorted(new_notes.keys()):
        notes_lines.append("==================================================")
        notes_lines.append(f"SLIDE {idx}")
        if idx in new_slides:
            # take first 80 chars of slide text as title hint
            title_hint = new_slides[idx][:80].strip()
            notes_lines.append(f"Titolo/Incipit: {title_hint}...")
        notes_lines.append(f"Note del relatore:\n\"{new_notes[idx]}\"")
        notes_lines.append("==================================================\n")
    with open(notes_dump_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(notes_lines))

    # 4. Compare with previous dump if exists
    if os.path.exists(old_dump_path):
        compare_dumps(old_dump_path, new_slides)
    else:
        print(f"[*] Nessun dump precedente trovato in {old_dump_path}. Il file corrente diventa il baseline.")

    # Update baseline dump
    with open(old_dump_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(out_lines))

    print("[*] Sincronizzazione completata con successo.")
    return True

def extract_from_pptx(pptx_file):
    slides_text = {}
    slides_notes = {}
    with zipfile.ZipFile(pptx_file, 'r') as z:
        # Read ordering
        pres_xml = z.read('ppt/presentation.xml')
        pres_root = ET.fromstring(pres_xml)
        rels_xml = z.read('ppt/_rels/presentation.xml.rels')
        rels_root = ET.fromstring(rels_xml)
        rel_map = {r.attrib.get('Id'): r.attrib.get('Target') for r in rels_root if r.attrib.get('Id')}

        slides_files = []
        for elem in pres_root.iter():
            if elem.tag.endswith('sldId'):
                r_id = elem.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
                if r_id and r_id in rel_map:
                    target = rel_map[r_id]
                    if not target.startswith('ppt/'):
                        target = 'ppt/' + target
                    slides_files.append(target)

        # Map slide to notesSlide
        slide_to_notes = {}
        for sfile in slides_files:
            sdir, sname = os.path.split(sfile)
            srels = f"{sdir}/_rels/{sname}.rels"
            if srels in z.namelist():
                srels_xml = z.read(srels)
                srels_root = ET.fromstring(srels_xml)
                for r in srels_root:
                    target = r.attrib.get('Target', '')
                    if 'notesSlide' in target:
                        norm_target = os.path.normpath(os.path.join(sdir, target)).replace('\\', '/')
                        slide_to_notes[sfile] = norm_target

        for idx, sfile in enumerate(slides_files, 1):
            s_xml = z.read(sfile)
            s_root = ET.fromstring(s_xml)
            texts = [node.text for node in s_root.iter() if node.tag.endswith('}t') and node.text]
            combined = re.sub(r'\s+', ' ', " ".join(texts)).strip()
            slides_text[idx] = combined

            if sfile in slide_to_notes:
                nfile = slide_to_notes[sfile]
                if nfile in z.namelist():
                    n_xml = z.read(nfile)
                    n_root = ET.fromstring(n_xml)
                    n_texts = [node.text for node in n_root.iter() if node.tag.endswith('}t') and node.text and node.text.strip() != str(idx)]
                    clean_n = re.sub(r'\s+', ' ', " ".join(n_texts)).strip()
                    if clean_n:
                        slides_notes[idx] = clean_n

    return slides_text, slides_notes

def compare_dumps(old_dump_file, new_slides):
    old_slides = {}
    current = None
    with open(old_dump_file, 'r', encoding='utf-8') as f:
        for line in f:
            m = re.match(r'^SLIDE\s+(\d+)', line.strip())
            if m:
                current = int(m.group(1))
                old_slides[current] = ""
            elif current is not None and not line.startswith('='):
                old_slides[current] += " " + line.strip()

    for k in old_slides:
        old_slides[k] = re.sub(r'\s+', ' ', old_slides[k]).strip()

    diffs = []
    max_count = max(len(old_slides), len(new_slides))
    for i in range(1, max_count + 1):
        ot = old_slides.get(i, "[NON ESISTEVA]")
        nt = new_slides.get(i, "[ELIMINATA]")
        if ot != nt:
            diffs.append((i, ot, nt))

    if not diffs:
        print("[+] Nessuna differenza rilevata rispetto all'ultimo dump locale.")
    else:
        print(f"[!] Rilevate differenze in {len(diffs)} slide:")
        for num, ot, nt in diffs:
            print(f"    - Slide {num}:")
            print(f"      PRIMA: {ot[:100]}...")
            print(f"      DOPO:  {nt[:100]}...")

if __name__ == "__main__":
    doc_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DOC_ID
    sync_presentation(doc_id)
