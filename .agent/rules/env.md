---
trigger: always_on
---

# Python & Environment Rules
- **Python Path:** "C:\Users\kouhi\anaconda3\python.exe"
- **Pip Path:** "C:\Users\kouhi\anaconda3\Scripts\pip.exe"
- **Execution:** Always use these absolute paths. Never use generic 'python' or 'pip' commands.
- **Memory Safety:** For 10GB .npy files in 'Demodata', ALWAYS use `mmap_mode='r'`.
- **Dependencies:** If a library is missing, use the Pip Path above to install it.

## Data & Schema
For rules regarding file headers, column names, and loading data, STRICTLY refer to `.agent/rules/data_schema.md` and the project's `dataset_config.json`.

## Coding rule
- Use the existing code and logic if possible.
- Don't put comment out just for memo, put doc string to explain the function is fine.