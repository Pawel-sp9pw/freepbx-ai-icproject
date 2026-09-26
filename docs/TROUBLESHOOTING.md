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
