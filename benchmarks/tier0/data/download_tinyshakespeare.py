"""Download TinyShakespeare dataset (1.1MB text file)."""

import os
import ssl
import urllib.request

try:
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except ImportError:
    pass

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(SAVE_DIR, "tinyshakespeare.txt")


def download():
    if os.path.exists(SAVE_PATH):
        print(f"TinyShakespeare already downloaded at {SAVE_PATH}")
        return SAVE_PATH

    print(f"Downloading TinyShakespeare from {URL}...")
    urllib.request.urlretrieve(URL, SAVE_PATH)
    size = os.path.getsize(SAVE_PATH)
    print(f"Downloaded {size / 1024:.1f} KB to {SAVE_PATH}")
    return SAVE_PATH


if __name__ == "__main__":
    download()
