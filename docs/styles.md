# Python Style Guide

## 1. Writing Great Python Code

### 1.1 Structuring Your Project

- Structure of the Repository: Keep the repository layout clear enough that users can quickly find source code, documentation, tests, packaging files, and licenses.
- Structure of Code is Key: Keep logic and dependencies explicit, and avoid circular dependencies, hidden coupling, global state, and deeply nested procedural code.
- Modules: Keep module names short, lowercase, importable, and focused on related data and functionality.
- Packages: Use packages to group related modules, and keep `__init__.py` files lightweight unless package-level exports are intentional.
- Object-oriented programming: Use classes when state and behavior naturally belong together, and prefer simpler functions when they do not.
- Decorators: Use decorators for reusable cross-cutting behavior that should stay separate from the decorated function's core logic.
- Context Managers: Use context managers when setup and cleanup should be paired reliably.
- Dynamic typing: Make dynamic code understandable through clear names, simple interfaces, and explicit validation where needed.
- Mutable and immutable types: Avoid unintended shared mutable state, especially across calls or object instances.

### 1.2 Code Style

- General concepts: Optimize for readability, consistency, and simple control flow.
- Idioms: Use Python idioms when they make intent clearer than patterns imported from other languages.
- Zen of Python: Prefer explicit, simple, and readable code over clever or overly abstract code.
- PEP 8: Follow PEP 8 unless a project-specific convention intentionally overrides it.
- Conventions: Match the existing local style before introducing a new convention.

### 1.3 Documentation

- Project Documentation: Keep user-facing documentation easy to find, with an entry point for setup and deeper references for details.
- Project Publication: Keep package metadata, license information, changelogs, and release documentation aligned before publishing.
- Code Documentation Advice: Use docstrings for public APIs and comments for implementation details that are not obvious from the code.

### 1.4 Testing Your Code

- The Basics: Write tests for behavior that should remain stable and for bugs that should not return.
- Tools: Choose testing tools that fit the project size, dependency model, and automation needs.

### 1.5 Logging

- ... or Print?: Use logging for diagnostics that need levels, configuration, or operational visibility.
- Logging in a Library: Libraries should create loggers but leave logging configuration to their users.
- Logging in an Application: Applications should configure handlers, formatters, levels, and destinations at the entry point.