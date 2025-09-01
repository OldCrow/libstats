# Versioning Strategy

## Current Status: Pre-1.0 Development

libstats is currently in active development toward a stable 1.0 release. During this pre-1.0 phase, we follow a modified semantic versioning approach.

## Pre-1.0 Versioning Rules (0.x.y)

While in 0.x versions:
- **0.x.0** - Minor version bumps for new features and breaking changes
- **0.x.y** - Patch version bumps for bug fixes and small improvements
- Breaking changes are documented but do NOT trigger major version bumps
- The API is considered unstable and may change between minor versions

## Commit Message Guidelines

During pre-1.0 development:
- Use `feat:` for new features → triggers 0.x.0 (minor bump)
- Use `fix:` for bug fixes → triggers 0.0.y (patch bump)
- Document breaking changes in commit body but avoid `BREAKING CHANGE:` footer
- Use `feat!:` or include breaking notes in PR descriptions

Example:
```
feat: add new distribution API

This update changes the distribution interface.
Note: The previous API is no longer supported.
```

## Post-1.0 Versioning (Future)

After reaching 1.0.0, we will follow strict semantic versioning:
- **Major (x.0.0)**: Breaking changes (triggered by `BREAKING CHANGE:`)
- **Minor (0.x.0)**: New features, backward compatible
- **Patch (0.0.x)**: Bug fixes, backward compatible

## Current Version

The project is at version **0.12.0** as of this documentation.

## Automated Releases

The GitHub Actions workflow uses semantic-release but is configured to:
- NOT trigger major version bumps from breaking changes while < 1.0
- Automatically update version numbers in CMakeLists.txt and include/libstats.h
- Generate changelogs using conventional commits

## Manual Version Override

If needed, versions can be manually set by:
1. Updating CMakeLists.txt (line ~143)
2. Updating include/libstats.h (lines ~176-179)
3. Creating an annotated git tag: `git tag -a v0.x.y -m "Release v0.x.y"`
