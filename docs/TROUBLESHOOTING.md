# Troubleshooting

## LXC powstał, ale port 8080 nie nasłuchuje

Jeżeli podczas tworzenia LXC pojawiło się:

```text
curl: (22) The requested URL returned error: 404
```

zaraz po:

```text
Customized LXC Container
```

to instalator aplikacji nie został uruchomiony.

Pierwsza wersja skryptu CT korzystała z upstreamowego `build.func`, który pobiera
instalatory wyłącznie z oficjalnego repo `community-scripts/ProxmoxVE`.
Problem został poprawiony.

Dla już utworzonego kontenera nie trzeba go kasować.

Na hoście Proxmox:

```bash
curl -fsSL https://raw.githubusercontent.com/Pawel-sp9pw/freepbx-ai-icproject/main/scripts/repair-existing-lxc.sh -o /tmp/repair-freepbx-ai.sh
bash /tmp/repair-freepbx-ai.sh 101
```

Po instalacji:

```bash
pct exec 101 -- systemctl status freepbx-ai --no-pager
pct exec 101 -- systemctl status piper-ai --no-pager
pct exec 101 -- systemctl status ollama --no-pager
pct exec 101 -- systemctl status caddy --no-pager
pct exec 101 -- ss -tlnp | grep -E ':8080 |:9019 '
```

Jeżeli któraś usługa nie działa:

```bash
pct exec 101 -- journalctl -u freepbx-ai -n 100 --no-pager
pct exec 101 -- journalctl -u piper-ai -n 100 --no-pager
pct exec 101 -- journalctl -u caddy -n 100 --no-pager
```


## Caddy: unrecognized directive basic_auth

Na Debian 13 pakiet systemowy może dostarczać Caddy 2.6.x. W tej wersji
dyrektywa `basic_auth` nie jest dostępna pod tą nazwą.

Projekt nie potrzebuje uwierzytelnienia w Caddy, ponieważ panel ma Basic Auth
zaimplementowany bezpośrednio w FastAPI.

Naprawa:

```bash
pct exec 101 -- bash -lc 'cat >/etc/caddy/Caddyfile <<EOF
:8080 {
    reverse_proxy 127.0.0.1:8000
}
EOF
caddy validate --config /etc/caddy/Caddyfile
systemctl restart caddy'
```

Następnie:

```bash
pct exec 101 -- ss -tlnp | grep -E ':8080 |:8000 |:9019 '
```
