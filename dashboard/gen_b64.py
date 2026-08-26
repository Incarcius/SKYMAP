import base64, sys, io, os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RGB_PATH = os.path.join(SCRIPT_DIR, '..', 'ml', 'data', 'RGB.png')
OUT_PATH = os.path.join(SCRIPT_DIR, 'src', 'data', 'rgb_b64.js')

with open(RGB_PATH, 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()

content = '// Auto-generated — base64 encoded orthophoto\n'
content += '// Source: ml/data/RGB.png\n'
content += 'export const IMG_B64 = "data:image/png;base64,' + b64 + '";\n'

with open(OUT_PATH, 'w', encoding='utf-8') as out:
    out.write(content)

print('rgb_b64.js written', len(b64) // 1024, 'KB')
