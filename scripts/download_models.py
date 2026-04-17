import os
import urllib.request

def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    urllib.request.urlretrieve(url, dest)
    print("Done!")

if __name__ == "__main__":
    os.makedirs("assets/models", exist_ok=True)
    inswapper_url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    
    dest = "assets/models/inswapper_128.onnx"
    if not os.path.exists(dest):
        download_file(inswapper_url, dest)
    else:
        print(f"Model {dest} already exists.")
