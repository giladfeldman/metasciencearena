"""Contract schemas, shipped as package data.

These JSON Schema files live INSIDE the package rather than at the repo root
because they are consumed by `framework.discovery` and `framework.storage` at
runtime — a wheel that validates against a schema it does not carry is not
installable. `contract/README.md` remains the prose specification and points
here; resolve a file with `framework.paths.schema_path(name)`, never by
composing a path from `__file__`.
"""
