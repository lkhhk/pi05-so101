"""Forward localhost:8002 to the cloud policy, without exposing a public model port."""
import argparse
import getpass
import os
import select
import socketserver

import paramiko


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True)
    p.add_argument('--ssh-port', type=int, required=True)
    p.add_argument('--user', default='root')
    p.add_argument('--local-port', type=int, default=8002)
    p.add_argument('--remote-port', type=int, default=8002)
    args = p.parse_args()
    password = os.environ.pop('PI05_SSH_PASSWORD', None) or getpass.getpass('SSH password: ')
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    # Connect only to a previously verified host; do not silently trust new keys.
    client.connect(args.host, port=args.ssh_port, username=args.user,
                   password=password, allow_agent=False, look_for_keys=False, timeout=20)
    del password
    transport = client.get_transport()
    transport.set_keepalive(20)

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            channel = transport.open_channel('direct-tcpip', ('127.0.0.1', args.remote_port),
                                             self.request.getpeername(), timeout=15)
            if channel is None:
                return
            try:
                while transport.is_active():
                    readable, _, _ = select.select([self.request, channel], [], [], 10)
                    for source in readable:
                        data = source.recv(65536)
                        if not data:
                            return
                        (channel if source is self.request else self.request).sendall(data)
            finally:
                channel.close()

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    try:
        with Server(('127.0.0.1', args.local_port), Handler) as server:
            print(f'Tunnel ready on 127.0.0.1:{args.local_port}', flush=True)
            server.serve_forever(poll_interval=0.5)
    finally:
        client.close()


if __name__ == '__main__':
    main()
