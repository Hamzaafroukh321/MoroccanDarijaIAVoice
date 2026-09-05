"""Fetch pinned portable experiment assets; no installation or inference.

Dry run by default. --download verifies exact lengths and SHA256 before use.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import zipfile

import httpx

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / '.local' / 'moulsot'
HF = 'https://huggingface.co/mradermacher/moulsot.v0.3-GGUF/resolve/ca66fea7f3db516212720fd005a0d70a57213de8/'
GH = 'https://github.com/ggml-org/llama.cpp/releases/download/b10809/'
ASSETS = [
    ('moulsot.v0.3.Q4_K_M.gguf', 1107405824, '9534be065f1990a0f8c4159ee16531cbefc5b0a1f9dcec47c57ea3088c3f9e78', HF),
    ('moulsot.v0.3.mmproj-Q8_0.gguf', 355709984, 'd530672efeaee3a0dad721334019e9155d2cea6184eeb126834fa6df2a8aa26c', HF),
    ('llama-b10809-bin-win-cuda-12.4-x64.zip', 253938543, 'c77bfcd9ed8d91e8721a2d6a290b907fddd4fa5412a47b21c6fa1709116b85f9', GH),
    ('cudart-llama-bin-win-cuda-12.4-x64.zip', 391443627, '8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6', GH),
]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(asset):
    name, size, sha, base = asset
    parent = DEST / ('models' if name.endswith('.gguf') else 'archives')
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / name
    if not (target.exists() and target.stat().st_size == size and digest(target) == sha):
        partial = target.with_suffix(target.suffix + '.partial')
        print(f'Downloading {name} ({size} bytes)', flush=True)
        downloaded = 0
        hasher = hashlib.sha256()
        with httpx.stream('GET', base + name, follow_redirects=True, timeout=60) as response:
            response.raise_for_status()
            with partial.open('wb') as stream:
                for chunk in response.iter_bytes(1024 * 1024):
                    downloaded += len(chunk)
                    if downloaded > size:
                        raise ValueError(f'{name}: exceeded pinned length')
                    hasher.update(chunk)
                    stream.write(chunk)
        if downloaded != size or hasher.hexdigest() != sha:
            raise ValueError(f'{name}: length or SHA256 mismatch; partial not used')
        partial.replace(target)
    print(f'Verified {name}', flush=True)
    if name.endswith('.zip'):
        extract_dir = DEST / 'runtime' / name.removesuffix('.zip')
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target) as archive:
            for member in archive.infolist():
                resolved = (extract_dir / member.filename).resolve()
                if not resolved.is_relative_to(extract_dir.resolve()):
                    raise ValueError('Archive path escapes experiment directory')
            archive.extractall(extract_dir)
        print(f'Extracted {name}', flush=True)
    return {'name': name, 'bytes': size, 'sha256': sha, 'url': base + name, 'path': str(target)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if not args.download:
        print(json.dumps({'bytes': sum(a[1] for a in ASSETS), 'assets': [a[0] for a in ASSETS]}, indent=2))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(prepare, ASSETS))
        (DEST / 'verified-assets.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
