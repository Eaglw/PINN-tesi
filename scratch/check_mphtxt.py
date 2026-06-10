import os
import sys

mphtxt_path = r"c:\Users\eaglw\Documents\PINN tesi\COMSOL\4roll\4_roll_mill_geom.mphtxt"

if not os.path.exists(mphtxt_path):
    print(f"File not found: {mphtxt_path}")
    sys.exit(1)

with open(mphtxt_path, 'r') as f:
    lines = [line.strip() for line in f]

# Parse edg2 elements
edg_elements = []
edg_entity_indices = []
edg_start = -1
for idx, line in enumerate(lines):
    if 'edg2 # type name' in line:
        edg_start = idx
        break

if edg_start != -1:
    num_edg_elements = 0
    edg_elem_idx = -1
    for i in range(edg_start, len(lines)):
        if '# number of elements' in lines[i]:
            num_edg_elements = int(lines[i].split('#')[0].strip())
            break
    for i in range(edg_start, len(lines)):
        if '# Elements' in lines[i]:
            edg_elem_idx = i + 1
            break
    
    if edg_elem_idx != -1:
        for i in range(num_edg_elements):
            parts = lines[edg_elem_idx + i].split()
            edg_elements.append([int(parts[0]), int(parts[1]), int(parts[2])])

    edg_entity_idx = -1
    for i in range(edg_elem_idx + num_edg_elements, len(lines)):
        if '# Geometric entity indices' in lines[i]:
            edg_entity_idx = i + 1
            break
    
    if edg_entity_idx != -1:
        for i in range(num_edg_elements):
            edg_entity_indices.append(int(lines[edg_entity_idx + i]))

# Parse Selections
selections = {}
idx = 0
while idx < len(lines):
    if 'Selection # class' in lines[idx]:
        label = ""
        for i in range(idx + 1, idx + 10):
            if '# Label' in lines[i]:
                raw_label = lines[i].split('#')[0].strip()
                parts = raw_label.split(maxsplit=1)
                if len(parts) == 2 and parts[0].isdigit():
                    label = parts[1]
                else:
                    label = raw_label
                break
        
        num_entities = 0
        ent_start = -1
        for i in range(idx + 1, idx + 20):
            if '# Number of entities' in lines[i]:
                num_entities = int(lines[i].split('#')[0].strip())
                ent_start = i + 2
                break
        
        entities = []
        if ent_start != -1:
            for i in range(num_entities):
                entities.append(int(lines[ent_start + i]))
        
        if label:
            selections[label] = entities
        idx = ent_start + num_entities
    else:
        idx += 1

print(f"Parsed {len(edg_elements)} edg2 elements.")
print(f"Unique geometric entity indices in edg2: {set(edg_entity_indices)}")
print("\nParsed Selections:")
for name, entities in selections.items():
    print(f"  - {name}: {entities}")
    # Check if entities overlap with edg_entity_indices
    overlap = set(entities).intersection(set(edg_entity_indices))
    print(f"    Overlap with edg_entity_indices: {overlap}")
