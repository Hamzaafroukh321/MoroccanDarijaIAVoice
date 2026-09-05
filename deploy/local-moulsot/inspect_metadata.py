"""Read GGUF scalar metadata without loading model tensors or importing torch."""
import json
import struct

from prepare import DEST


def inspect(path):
    with path.open('rb') as stream:
        def unpack(fmt):
            return struct.unpack('<' + fmt, stream.read(struct.calcsize('<' + fmt)))[0]

        def string():
            length = unpack('Q')
            if length > 1024 * 1024:
                raise ValueError('Unexpected metadata string size')
            return stream.read(length).decode('utf-8')

        def value(kind):
            formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
            if kind in formats:
                return unpack(formats[kind])
            if kind == 8:
                return string()
            if kind == 9:
                subtype, count = unpack('I'), unpack('Q')
                if count > 1024 * 1024:
                    raise ValueError('Unexpected metadata array size')
                for _ in range(count):
                    value(subtype)
                return f'array[{count}]'
            raise ValueError(f'Unknown GGUF type {kind}')

        if stream.read(4) != b'GGUF':
            raise ValueError('Not a GGUF file')
        version, tensors, keys = unpack('I'), unpack('Q'), unpack('Q')
        result = {'file': path.name, 'version': version, 'tensor_count': tensors, 'metadata_count': keys, 'metadata': {}}
        if version != 3 or keys > 10000:
            raise ValueError('Unexpected GGUF header')
        for _ in range(keys):
            key = string()
            entry = value(unpack('I'))
            if key.startswith('clip.') or (key.startswith(('general.', 'qwen3vl.')) and not isinstance(entry, str)) or key in {
                'general.architecture', 'general.name', 'clip.projector_type'}:
                result['metadata'][key] = entry
        return result


if __name__ == '__main__':
    print(json.dumps([inspect(path) for path in sorted((DEST / 'models').glob('*.gguf'))], indent=2))
