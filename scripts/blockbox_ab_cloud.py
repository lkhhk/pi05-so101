"""Authenticate in memory, inspect/deploy the baseline, and optionally forward its port."""
import argparse
import getpass
import json
from pathlib import Path
import select
import socketserver
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'vendor/cloud_ssh'))
import paramiko


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--password-stdin',action='store_true')
    p.add_argument('--deploy',action='store_true')
    p.add_argument('--forward',action='store_true')
    p.add_argument('--command',help='Run a quoted SSH command after inspection')
    args=p.parse_args()
    password=input() if args.password_stdin else getpass.getpass('Cloud SSH password: ')
    client=paramiko.SSHClient();client.load_system_host_keys()
    client.connect('your-gpu-host.example.com',port=2222,username='root',password=password,
                   allow_agent=False,look_for_keys=False,timeout=15)
    del password
    def command(cmd):
        _,stdout,stderr=client.exec_command(cmd)
        output=stdout.read().decode();error=stderr.read().decode()
        print(output,flush=True)
        if error:print(error,file=sys.stderr,flush=True)
        status=stdout.channel.recv_exit_status()
        if status:raise RuntimeError(f'Remote command exited {status}')
    try:
        command("nvidia-smi --query-gpu=memory.used,memory.free --format=csv; pgrep -af 'serve_blockbox|lerobot.train|lerobot-train' || true")
        command("cat /root/autodl-tmp/so101-block-box-lora-run-20261117/checkpoints/020000/pretrained_model/adapter_config.json")
        if args.command:command(args.command)
        if args.deploy:
            with client.open_sftp() as sftp:
                for name in ('serve_blockbox_ab.py','blockbox_ab_protocol.py','start_blockbox_ab_server.sh'):
                    sftp.put(str(ROOT/'scripts'/name),'/root/autodl-tmp/'+name)
            command('bash /root/autodl-tmp/start_blockbox_ab_server.sh')
        if not args.forward:return
        transport=client.get_transport();transport.set_keepalive(20)
        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                channel=transport.open_channel('direct-tcpip',('127.0.0.1',8093),self.request.getpeername(),timeout=15)
                if channel is None:return
                try:
                    while transport.is_active():
                        readable,_,_=select.select([self.request,channel],[],[],10)
                        for source in readable:
                            data=source.recv(65536)
                            if not data:return
                            (channel if source is self.request else self.request).sendall(data)
                finally:channel.close()
        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address=True
            daemon_threads=True
        with Server(('127.0.0.1',8093),Handler) as server:
            print('BASELINE_TUNNEL_READY 127.0.0.1:8093',flush=True)
            server.serve_forever()
    finally:client.close()


if __name__=='__main__':main()
