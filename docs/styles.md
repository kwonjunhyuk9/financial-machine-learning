# Style Guide

## 1. Writing Great Python Code

### 1.1 Structuring Your Project

- Keep the repository layout clear enough that users can quickly find source code, documentation, tests, packaging
  files, and licenses.
- Keep logic and dependencies explicit, and avoid circular dependencies, hidden coupling,
  global state, and deeply nested procedural code.
- Keep module names short, lowercase, importable, and focused on related data and functionality.
- Use packages to group related modules, and keep `__init__.py` files lightweight unless package-level exports
  are intentional.
- Use classes when state and behavior naturally belong together, and prefer simpler
  functions when they do not.
- Use decorators for reusable cross-cutting behavior that should stay separate from the decorated function's
  core logic.
- Use context managers when setup and cleanup should be paired reliably.
- Make dynamic code understandable through clear names, simple interfaces, and explicit validation where
  needed.
- Avoid unintended shared mutable state, especially across calls or object instances.

### 1.2 Code Style

- Optimize for readability, consistency, and simple control flow.
- Use Python idioms when they make intent clearer than patterns imported from other languages.
- Prefer explicit, simple, and readable code over clever or overly abstract code.
- Follow PEP 8 unless a project-specific convention intentionally overrides it.
- Match the existing local style before introducing a new convention.

### 1.3 Documentation

- Keep user-facing documentation easy to find, with an entry point for setup and deeper references for details.
- Keep package metadata, license information, changelogs, and release documentation aligned before publishing.
- Use docstrings for public APIs and comments for implementation details that are not obvious
  from the code.

### 1.4 Testing Your Code

- Write tests for behavior that should remain stable and for bugs that should not return.
- Choose testing tools that fit the project size, dependency model, and automation needs.

### 1.5 Logging

- Use logging for diagnostics that need levels, configuration, or operational visibility.
- Libraries should create loggers but leave logging configuration to their users.
- Applications should configure handlers, formatters, levels, and destinations at the entry point.

## 2. Notebook Explanation Guidelines

### 2.1 Structure

- Start the notebook with a single `#` title written in title case.
- Add `## Overview` immediately below the notebook title and use that heading exactly.
- Begin the Overview with one sentence explaining which module or public functions the notebook demonstrates and what
  example or dataset it uses.
- Include only `Problem` and `Approach` as bullet points in the Overview.
- Follow the common section flow: `Overview` → `Synthetic Data` or `Observed Data` → computation → output →
  visualization or interpretation.
- Use `## Synthetic Data` exactly when the notebook creates its own example, simulated, or generated data.
- Use `## Observed Data` exactly when the notebook fetches or loads historical, market, fundamental, news, or other
  externally observed data.
- Include both fixed data headings in source order when a notebook uses both synthetic and observed data.
- Place each explanatory Markdown cell immediately before the code cell it describes.
- Except for the Overview, begin every explanatory Markdown cell with a descriptive ## heading.
- After the fixed data heading, use headings that describe the domain computation or result, such as
  `## Probability-Based Bet Sizing`, rather than procedural headings such as `## Run Code`.
- Keep computation, output, visualization, and interpretation sections in execution order, omitting only stages that
  have no distinct explanatory or code cell.

### 2.2 Content

- Begin each later explanatory cell with one sentence stating the cell’s semantic purpose.
- Use verbs such as defines, computes, reports, fetches, saves, or visualizes.
- Describe what the cell produces or demonstrates rather than listing implementation steps such as “calls,” “reads,” or
  “inspects.”
- Describe each key input variable, table row, and table column as an individual bullet point.
- Describe each key output value, table row, and table column as an individual bullet point.
- Present input bullets in the same order that the inputs appear in the code.
- Present output bullets in the same order that the values or columns appear in the displayed result.__
- Explain the semantic role of each item, not merely its Python type or literal value.
- State units, time zones, frequencies, sign conventions, valid categorical values, and index meaning whenever they
  affect interpretation.
- Explain relationships between fields when an output is derived from other values.
- Distinguish source data, derived features, model outputs, and evaluation statistics explicitly.
- Source and output directories may be named explicitly, but describe individual files neutrally without naming
  specific files.
- Keep dataset references neutral in explanatory cells; do not mention specific identifiers such as ticker symbols,
  even when the current dataset uses them.
- For synthetic data, describe the assumptions used to generate it when they affect the interpretation of results.
- For plots, describe the axes, visual encodings, and main interpretation rather than only stating that the cell draws a
  chart.
- When a result demonstrates a specific behavior, explain the observed takeaway without making claims beyond the
  displayed result.

### 2.3 Formatting and Terminology

- Write all notebook explanations in English.
- Write explanations as complete sentences ending with periods.
- Use one hyphenated bullet per described item.
- Avoid nested bullets unless an item genuinely requires multiple distinct conditions.
- Enclose modules, functions, classes, variables, parameters, column names, index names, literal values, and file paths
  in backticks.
- Use the same identifier spelling and terminology as the corresponding source code and displayed output.
- Use descriptive names consistently across the heading, explanation, code, and output.
- Expand an uncommon abbreviation at its first meaningful use, then use the abbreviation consistently.
- Do not use bold emphasis as a substitute for headings or identifier formatting.
- Do not mention AFML anywhere in the explanations.

### 2.4 Scope and Maintenance

- Explain public behavior and data meaning rather than private implementation details.
- Do not narrate obvious code operations line by line.
- Do not document variables, rows, or columns that are not present in the corresponding cell.
- Update the explanation whenever an input, output schema, plot, or execution order changes.
- Keep each explanation limited to the cell immediately following it.
- Avoid repeating information already defined in an earlier section unless its meaning changes in the new context.
- Keep fetch notebooks parallel by using `## Observed Data` for the request and source description, followed by the
  relevant output section.
- Keep related notebooks consistent in sentence patterns, bullet wording, capitalization, and section ordering.
