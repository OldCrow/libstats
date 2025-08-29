#!/bin/bash

# Fix all cpu:: namespace references and arch:: constant references in SIMD implementation files

echo "Fixing SIMD namespace references..."

# List of SIMD implementation files to fix
SIMD_FILES=(
    "src/simd_sse2.cpp"
    "src/simd_avx.cpp"
    "src/simd_avx2.cpp"
    "src/simd_avx512.cpp"
    "src/simd_dispatch.cpp"
    "src/simd_neon.cpp"
    "src/simd_fallback.cpp"
)

for file in "${SIMD_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Processing $file..."
        
        # Fix cpu:: references - the functions are in stats::arch namespace
        # Change cpu::supports_xxx() to just supports_xxx() since we're already in stats::arch
        sed -i '' 's/cpu::supports_sse2()/supports_sse2()/g' "$file"
        sed -i '' 's/cpu::supports_sse4_1()/supports_sse4_1()/g' "$file"
        sed -i '' 's/cpu::supports_avx()/supports_avx()/g' "$file"
        sed -i '' 's/cpu::supports_avx2()/supports_avx2()/g' "$file"
        sed -i '' 's/cpu::supports_avx512()/supports_avx512()/g' "$file"
        sed -i '' 's/cpu::supports_fma()/supports_fma()/g' "$file"
        sed -i '' 's/cpu::supports_neon()/supports_neon()/g' "$file"
        
        # Fix arch:: constant references - they should be arch::simd::
        sed -i '' 's/arch::SSE_DOUBLES/arch::simd::SSE_DOUBLES/g' "$file"
        sed -i '' 's/arch::AVX_DOUBLES/arch::simd::AVX_DOUBLES/g' "$file"
        sed -i '' 's/arch::AVX2_DOUBLES/arch::simd::AVX2_DOUBLES/g' "$file"
        sed -i '' 's/arch::AVX512_DOUBLES/arch::simd::AVX512_DOUBLES/g' "$file"
        sed -i '' 's/arch::NEON_DOUBLES/arch::simd::NEON_DOUBLES/g' "$file"
        
        # Also fix any other arch:: constants that might be in simd namespace
        sed -i '' 's/arch::SSE_ALIGNMENT/arch::simd::SSE_ALIGNMENT/g' "$file"
        sed -i '' 's/arch::AVX_ALIGNMENT/arch::simd::AVX_ALIGNMENT/g' "$file"
        sed -i '' 's/arch::AVX512_ALIGNMENT/arch::simd::AVX512_ALIGNMENT/g' "$file"
        sed -i '' 's/arch::NEON_ALIGNMENT/arch::simd::NEON_ALIGNMENT/g' "$file"
    fi
done

echo "SIMD namespace fixes complete!"
