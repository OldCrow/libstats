#!/bin/zsh
# Kaby Lake (AVX2) build of the tools/bench comparatives: v2.4.1 worktree vs dev/v2.5.0-corvus.
# Binaries land in $NB/tests/ so corvus tools/quiet_bench.sh can run them from the Release tree.
set -eu
OLD=/Users/wolfman/Development/libstats-v2.4.1
NEW=/Users/wolfman/Development/libstats
NB=$NEW/build-bench
CORVUS_SRC=$NB/_deps/corvus-src
CORVUS_LIB=$NB/_deps/corvus-build/libcorvus.a
HWY_LIB=/usr/local/lib
OUT=$NB/tests
CXX="clang++ -std=c++20 -O2 -DNDEBUG"
cd $NEW/tools/bench
$CXX -I$OLD/include -I$OLD/include/libstats -I$OLD/build/generated distributions_bench.cpp $OLD/build/libstats.a -lpthread -o $OUT/dist_v241
$CXX -I$OLD/include -I$OLD/include/libstats -I$OLD/build/generated elementary_bench.cpp    $OLD/build/libstats.a -lpthread -o $OUT/elem_v241
$CXX -I$NEW/include -I$NEW/include/libstats -I$NB/generated distributions_bench.cpp $NB/libstats.a $CORVUS_LIB -L$HWY_LIB -lhwy -lhwy_contrib -lpthread -o $OUT/dist_v250
$CXX -I$NEW/include -I$NEW/include/libstats -I$NB/generated elementary_bench.cpp    $NB/libstats.a $CORVUS_LIB -L$HWY_LIB -lhwy -lhwy_contrib -lpthread -o $OUT/elem_v250
$CXX -I$CORVUS_SRC/include corvus_scaling_bench.cpp $CORVUS_LIB -L$HWY_LIB -lhwy -lhwy_contrib -lpthread -o $OUT/corvus_scaling
ls -l $OUT/dist_v241 $OUT/elem_v241 $OUT/dist_v250 $OUT/elem_v250 $OUT/corvus_scaling
