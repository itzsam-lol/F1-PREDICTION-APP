import re, ast

content = open('c:/Users/Satyam/Desktop/f1-prediction-app/app.py', encoding='utf-8').read()
lines = content.splitlines()

# Check for 8-digit hex color patterns that Plotly rejects
bad_patterns = [r'gridcolor="#[0-9a-fA-F]{8}"', r'zerolinecolor="#[0-9a-fA-F]{8}"',
                r'fillcolor=tc', r'linecolor="#[0-9a-fA-F]{8}"',
                r'gridcolor="#ffffff', r'zerolinecolor="#ffffff']
found = []
for i, line in enumerate(lines):
    for pat in bad_patterns:
        if re.search(pat, line):
            found.append((i+1, line.strip()))

if found:
    print("BAD COLORS FOUND:")
    for ln, l in found:
        print(f"  L{ln}: {l}")
else:
    print("No invalid Plotly color patterns found!")

# Syntax check
try:
    ast.parse(content)
    print("Syntax OK!")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
