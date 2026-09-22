#!/usr/bin/env python3
"""Insert the /pawn/ reverse-proxy location into /etc/nginx/nginx.conf.

Run with sudo from the repo root:
    sudo python3 deploy/apply_nginx_pawn_location.py
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

NGINX_CONF = Path("/etc/nginx/nginx.conf")

NEEDLE = """    location / {
        proxy_pass http://127.0.0.1:4001;"""

INSERT = """    # pawn-server — keep uvicorn on 127.0.0.1:8000 only
    location ^~ /pawn/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header X-Forwarded-Host $host:$server_port;
        proxy_set_header Authorization $http_authorization;

        # chat streaming + long agent turns
        proxy_buffering off;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        client_max_body_size 100m;
    }

"""


def main() -> None:
    text = NGINX_CONF.read_text()
    if "location ^~ /pawn/" in text:
        print("Already present — nothing to do.")
        return
    if NEEDLE not in text:
        raise SystemExit("Could not find the :4001 location / block to insert before.")

    backup = NGINX_CONF.with_name(
        f"nginx.conf.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    shutil.copy2(NGINX_CONF, backup)
    NGINX_CONF.write_text(text.replace(NEEDLE, INSERT + NEEDLE, 1))
    print(f"Updated {NGINX_CONF}")
    print(f"Backup: {backup}")
    print("Next: nginx -t && systemctl reload nginx")


if __name__ == "__main__":
    main()
