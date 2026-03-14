# Tests

Recommended layout:

- `tests/experiments/`
- `tests/plugins/`
- `tests/strategies/`
- `tests/ui/`

Priority smoke coverage to add first:

- limited spec building and monkey seed determinism
- result loader behavior for incomplete and complete run folders
- strategy discovery and plugin registry loading
- workbook scenario defaults and validation helpers
