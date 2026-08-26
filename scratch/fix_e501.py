import re

def fix_ui_styles():
    with open('src/ui_styles.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open('src/ui_styles.py', 'w', encoding='utf-8') as f:
        for line in lines:
            if len(line) > 89:
                line = line.replace('"font_sans": "-apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Helvetica, Arial, sans-serif",', '\"font_sans\": (\n        \"-apple-system, BlinkMacSystemFont, \"\n        \"\'Segoe UI\', Roboto, Helvetica, Arial, sans-serif\"\n    ),')
                line = line.replace("font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;", "font-family: -apple-system, BlinkMacSystemFont,\n        'Segoe UI', Roboto, Helvetica, Arial, sans-serif;")
                
                if '{' in line and '}' in line and len(line) > 88:
                    rule_match = re.match(r'^(.*?)\s*\{\s*(.*?)\s*\}\s*$', line)
                    if rule_match:
                        sel, styles = rule_match.groups()
                        styles_split = styles.replace('; ', ';\n    ')
                        line = f'{sel} {{\n    {styles_split}\n}}\n'
            f.write(line)

def fix_ui_components():
    with open('src/ui_components.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open('src/ui_components.py', 'w', encoding='utf-8') as f:
        for line in lines:
            # We can split f'<div style="...">...' into multiple strings or use f"""..."""
            if 'render_html(' in line or 'f"<div' in line or "f'<div" in line or '<div' in line:
                if len(line) > 89:
                    line = re.sub(r'style="([^"]{50,})"', lambda m: 'style="\n    ' + m.group(1).replace(';', ';\n    ') + '"', line)
            f.write(line)
            
if __name__ == '__main__':
    fix_ui_styles()
    fix_ui_components()
