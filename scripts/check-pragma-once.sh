#!/bin/bash

# Check that all header files use #pragma once
# This is the libstats convention for header guards

EXIT_CODE=0

for file in "$@"; do
    # Skip if file doesn't exist (might be deleted)
    [ -f "$file" ] || continue

    # Check if the file contains #pragma once
    if ! grep -q "^#pragma once" "$file"; then
        echo "ERROR: $file is missing '#pragma once'"
        echo "       libstats uses #pragma once as the header guard convention"
        EXIT_CODE=1
    fi
done

exit $EXIT_CODE
