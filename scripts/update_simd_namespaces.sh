#!/bin/bash

# Phase 3B: Update SIMD implementation files to use new namespace structure

echo "Updating SIMD implementation files to use new namespace structure..."

# List of SIMD implementation files that need updating
FILES=(
    "src/simd_neon.cpp"
    "src/simd_avx.cpp"
    "src/simd_avx2.cpp"
    "src/simd_avx512.cpp"
    "src/simd_sse2.cpp"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Updating $file..."

        # Replace namespace declarations
        sed -i.bak -e 's/namespace stats {/namespace stats {/g' \
                   -e 's/namespace arch {/namespace simd {/g' \
                   -e 's/namespace simd {$/namespace ops {/g' \
                   -e 's/}  \/\/ namespace simd$/}  \/\/ namespace ops/g' \
                   -e 's/}  \/\/ namespace arch$/}  \/\/ namespace simd/g' \
                   "$file"

        # Fix any double namespace simd issues
        sed -i.bak2 -e '/namespace simd {/,/namespace ops {/{
            s/namespace simd {//2
        }' "$file"

        # Clean up backup files
        rm -f "${file}.bak" "${file}.bak2"

        echo "  ✓ Updated $file"
    else
        echo "  ⚠ File not found: $file"
    fi
done

echo "Done! All SIMD implementation files have been updated."
echo "Please rebuild the project to verify the changes."
