#!/usr/bin/env python3

import os
import re

def fix_example_file(filepath):
    """Fix an example file to use LIBSTATS_FULL_INTERFACE"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if 'LIBSTATS_FULL_INTERFACE' in content:
        print(f"✓ {os.path.basename(filepath)} already fixed")
        return False
    
    # Replace the libstats.h include line
    pattern = r'#include "libstats\.h"'
    replacement = '#define LIBSTATS_FULL_INTERFACE\n#include "libstats.h"'
    
    if re.search(pattern, content):
        new_content = re.sub(pattern, replacement, content)
        
        # Make sure we have iostream if std::cout is used
        if 'std::cout' in content and '#include <iostream>' not in content:
            # Add iostream after the libstats include
            new_content = new_content.replace(
                '#include "libstats.h"', 
                '#include "libstats.h"\n#include <iostream>'
            )
        
        # Make sure we have random if std::mt19937 is used
        if 'std::mt19937' in content and '#include <random>' not in content:
            new_content = new_content.replace(
                '#include <iostream>', 
                '#include <iostream>\n#include <random>'
            )
        
        # Make sure we have iomanip if std::setprecision is used
        if 'std::setprecision' in content and '#include <iomanip>' not in content:
            new_content = new_content.replace(
                '#include <random>', 
                '#include <random>\n#include <iomanip>'
            )
        elif 'std::setprecision' in content and '#include <iostream>' in content:
            new_content = new_content.replace(
                '#include <iostream>', 
                '#include <iostream>\n#include <iomanip>'
            )
        
        # Make sure we have vector if std::vector is used
        if 'std::vector' in content and '#include <vector>' not in content:
            # Find the last standard include and add vector after it
            lines = new_content.split('\n')
            last_std_include_idx = -1
            for i, line in enumerate(lines):
                if line.startswith('#include <') and not line.startswith('#include <iostream>'):
                    last_std_include_idx = i
            
            if last_std_include_idx >= 0:
                lines.insert(last_std_include_idx + 1, '#include <vector>')
            elif '#include <iostream>' in new_content:
                new_content = new_content.replace(
                    '#include <iostream>', 
                    '#include <iostream>\n#include <vector>'
                )
            new_content = '\n'.join(lines)
        
        with open(filepath, 'w') as f:
            f.write(new_content)
        
        print(f"✓ Fixed {os.path.basename(filepath)}")
        return True
    else:
        print(f"⚠ No libstats.h include found in {os.path.basename(filepath)}")
        return False

def main():
    examples_dir = 'examples'
    fixed_count = 0
    
    print("🔧 Fixing examples to use LIBSTATS_FULL_INTERFACE...")
    
    for filename in os.listdir(examples_dir):
        if filename.endswith('.cpp'):
            filepath = os.path.join(examples_dir, filename)
            if fix_example_file(filepath):
                fixed_count += 1
    
    print(f"\n✅ Fixed {fixed_count} example files")
    print("\nAll examples should now compile with the Phase 1 header optimization!")

if __name__ == "__main__":
    main()
