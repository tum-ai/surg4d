#!/usr/bin/env python3
"""
Fix the notebook by replacing all general_processor references with consistent_processor
"""

import json

def fix_notebook():
    """Fix the notebook file"""
    
    # Read the notebook
    with open('dev_fabian copy.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    changes_made = 0
    
    # Go through all cells and replace general_processor with consistent_processor
    for cell in notebook['cells']:
        if 'source' in cell and cell['cell_type'] == 'code':
            # Convert source to string if it's a list
            if isinstance(cell['source'], list):
                source_str = ''.join(cell['source'])
            else:
                source_str = cell['source']
            
            # Check if this cell contains general_processor
            if 'general_processor' in source_str:
                print(f"Found general_processor in cell, fixing...")
                
                # Replace all occurrences
                new_source = source_str.replace('general_processor', 'consistent_processor')
                
                # Convert back to list format if needed
                if isinstance(cell['source'], list):
                    cell['source'] = [new_source]
                else:
                    cell['source'] = new_source
                
                changes_made += 1
                print(f"  Replaced general_processor with consistent_processor")
    
    # Save the fixed notebook
    with open('dev_fabian copy.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"\n✓ Fixed notebook with {changes_made} changes")
    print("✓ All general_processor references replaced with consistent_processor")
    
    return changes_made

if __name__ == "__main__":
    print("=== FIXING NOTEBOOK ===")
    changes = fix_notebook()
    print(f"=== COMPLETED: {changes} changes made ===")
