from PIL import Image, ImageDraw
import os

os.makedirs('alpha_backtester/pwa/icons', exist_ok=True)

def create_icon(size, path):
    img = Image.new('RGBA', (size, size), (15, 23, 42, 255))  # Dark slate background
    draw = ImageDraw.Draw(img)

    c_green = (0, 255, 163, 255)
    c_cyan = (0, 184, 217, 255)

    # Wick 1
    w1_x = size * 0.35
    draw.line([(w1_x, size * 0.2), (w1_x, size * 0.8)], fill=c_green, width=max(2, int(size * 0.03)))
    # Body 1
    b1_w = size * 0.16
    draw.rectangle([w1_x - b1_w/2, size * 0.3, w1_x + b1_w/2, size * 0.65], fill=c_green)

    # Wick 2
    w2_x = size * 0.65
    draw.line([(w2_x, size * 0.15), (w2_x, size * 0.75)], fill=c_cyan, width=max(2, int(size * 0.03)))
    # Body 2
    b2_w = size * 0.16
    draw.rectangle([w2_x - b2_w/2, size * 0.22, w2_x + b2_w/2, size * 0.52], fill=c_cyan)

    img.save(path)
    print(f"Saved {path}")

create_icon(192, 'alpha_backtester/pwa/icons/icon-192.png')
create_icon(512, 'alpha_backtester/pwa/icons/icon-512.png')
create_icon(64, 'alpha_backtester/pwa/icons/favicon.png')
