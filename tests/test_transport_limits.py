"""Ephemeral loopback tests; no application providers or persistent servers."""

import asyncio
import random
import socket

import pytest
import uvicorn
import websockets
from websockets.exceptions import ConnectionClosedError


async def exchange(payload, *, server_compression, client_compression, max_size=1024, max_queue=32):
    received = []

    async def app(scope, receive, send):
        assert scope['type'] == 'websocket'
        assert (await receive())['type'] == 'websocket.connect'
        await send({'type': 'websocket.accept'})
        message = await receive()
        if message['type'] == 'websocket.receive':
            received.append(message['bytes'])
            await send({'type': 'websocket.send', 'text': str(len(message['bytes']))})
            await send({'type': 'websocket.close', 'code': 1000})

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(('127.0.0.1', 0))
    listener.listen(8)
    port = listener.getsockname()[1]
    config = uvicorn.Config(app, host='127.0.0.1', port=port, ws='websockets',
                            lifespan='off', ws_max_size=max_size, ws_max_queue=max_queue,
                            ws_per_message_deflate=server_compression,
                            log_config=None, log_level='critical')
    server = uvicorn.Server(config)
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if serving.done():
                    await serving
                    raise AssertionError('The local test server stopped before startup.')
                await asyncio.sleep(.01)
            async with websockets.connect(f'ws://127.0.0.1:{port}/', compression=client_compression,
                                          open_timeout=2, close_timeout=1) as connection:
                negotiated = bool(connection.extensions)
                await connection.send(payload)
                try:
                    acknowledgement = await asyncio.wait_for(connection.recv(), 2)
                    return {'received': received, 'acknowledgement': acknowledgement,
                            'code': 1000, 'compressed': negotiated}
                except ConnectionClosedError as exc:
                    return {'received': received, 'acknowledgement': None,
                            'code': exc.rcvd.code if exc.rcvd else None, 'compressed': negotiated}
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(serving, 3)
        finally:
            if not serving.done():
                serving.cancel()
                await asyncio.gather(serving, return_exceptions=True)
            listener.close()


@pytest.mark.parametrize('server_compression,client_compression,entropy,size,expected', [
    (True, 'deflate', False, 1024, 1000),
    (True, 'deflate', True, 1024, 1009),
    (False, 'deflate', True, 1024, 1000),
    (True, None, True, 1024, 1000),
    (False, 'deflate', True, 1025, 1009),
])
def test_pcm_boundary_with_and_without_websocket_compression(server_compression, client_compression, entropy, size, expected):
    payload = random.Random(20260905).randbytes(size) if entropy else bytes(size)
    result = asyncio.run(exchange(payload, server_compression=server_compression,
                                  client_compression=client_compression))
    assert result['code'] == expected
    assert result['compressed'] is (server_compression and client_compression is not None)
    if expected == 1000:
        assert result['received'] == [payload]
        assert result['acknowledgement'] == str(size)
    else:
        assert result['received'] == []
        assert result['acknowledgement'] is None


def test_production_server_transport_settings_accept_pcm_and_reject_oversize(monkeypatch):
    from engine import server as production

    captured = {}

    def capture_run(app, **kwargs):
        assert app is production.app
        captured.update(kwargs)

    monkeypatch.setattr(production.uvicorn, 'run', capture_run)
    production.run_server()
    assert captured['ws_per_message_deflate'] is False
    assert captured['ws_max_size'] == 1024
    assert captured['ws_max_queue'] == production.runtime['max_buffered_frames']

    for size, expected in ((1024, 1000), (1025, 1009)):
        payload = random.Random(20260905).randbytes(size)
        result = asyncio.run(exchange(
            payload, server_compression=captured['ws_per_message_deflate'],
            client_compression='deflate', max_size=captured['ws_max_size'],
            max_queue=captured['ws_max_queue'],
        ))
        assert result['code'] == expected
        assert result['compressed'] is False
        assert result['received'] == ([payload] if expected == 1000 else [])
