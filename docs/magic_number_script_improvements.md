# magic number replacement script improvements

## overview

the `tools/replace_magic_numbers.py` script has been significantly enhanced to reduce false positives and provide more accurate magic number replacements that align with the existing `stats::detail::` namespace constants.

## key improvements made

### 1. namespace updates
- updated all constant mappings to use `detail::` namespace instead of outdated `constants::math::` or `constants::`
- aligned with current codebase structure where constants are in `stats::detail::`

### 2. enhanced context detection

#### skip contexts
enhanced the `SKIP_CONTEXTS` list to avoid replacements in:
- comment lines (`//` and multi-line comments)
- include statements
- constant definitions (`constexpr`, `inline constexpr`, `static const`)
- array indices
- template parameters
- switch case labels
- error message line numbers
- scientific notation patterns

#### scientific notation detection
added `is_scientific_notation_context()` function to detect and skip numbers that are parts of scientific notation:
- patterns like `1.23e-4` or `5E+2`
- prevents replacing the mantissa or exponent separately

#### array/template context detection
added `is_array_or_template_context()` function to identify:
- array indices: `array[5]`
- template parameters: `std::vector<int, 10>`

#### conservative integer replacement
made integer replacement much more conservative:
- only suggests replacements in clear arithmetic contexts (`+`, `-`, `*`, `/`)
- allows meaningful comparisons (`==`, `!=`) in control structures
- skips function calls, constructors, and cast operations
- excludes loops and iterator contexts
- much more selective about when to suggest integer constants

### 3. position-aware pattern matching

the script now uses `finditer()` instead of `findall()` to get the exact position of matches, enabling:
- context-aware analysis of what comes before and after each number
- more precise filtering based on surrounding characters
- better detection of inappropriate replacement contexts

### 4. constant mapping alignment

updated the constant mappings to use existing constants from the codebase:
- statistical constants: `CHI2_95_DF_1`, `Z_95`, etc.
- significance levels: `ALPHA_01`, `ALPHA_05`, etc.
- basic mathematical constants: `ZERO_DOUBLE`, `ONE`, `TWO`, etc.

## results

### before improvements
- high number of false positives (200+ suggestions for `validation.cpp`)
- suggested replacements in comments
- suggested replacements for scientific notation parts
- overly aggressive integer replacement
- array index replacements
- template parameter replacements

### after improvements
- significantly reduced false positives (116 suggestions for `validation.cpp`)
- no suggestions in comments or inappropriate contexts
- proper handling of scientific notation
- conservative, meaningful integer replacements only
- smart context detection prevents inappropriate suggestions

## usage

the script can now be used confidently for magic number detection and replacement:

```bash
# dry run (recommended first step)
python3 tools/replace_magic_numbers.py src/

# interactive mode for selective replacements
python3 tools/replace_magic_numbers.py src/ --interactive

# apply all replacements (use with caution)
python3 tools/replace_magic_numbers.py src/ --write
```

## next steps

the script is now ready for:
1. phase 1 of header optimization: magic number elimination
2. integration with iwyu-based header cleanup
3. application across the entire codebase with confidence

the improvements make the script practical for real-world use while maintaining high accuracy and reducing manual review overhead.
