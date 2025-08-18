#pragma once

#if defined(_WIN32) || defined(_WIN64)
  #if defined(LIBSTATS_BUILDING_SHARED)
    #define LIBSTATS_API __declspec(dllexport)
  #elif defined(LIBSTATS_USING_SHARED)
    #define LIBSTATS_API __declspec(dllimport)
  #else
    #define LIBSTATS_API
  #endif
#else
  #if defined(__GNUC__) || defined(__clang__)
    #define LIBSTATS_API __attribute__((visibility("default")))
  #else
    #define LIBSTATS_API
  #endif
#endif

