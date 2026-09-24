# FreePBX AI → IC Project (LXC / Proxmox)

Self-hosted voice agent dla małej firmy:

- FreePBX / Asterisk
- Asterisk AudioSocket (dwukierunkowe audio po TCP)
- faster-whisper (STT)
- Ollama (lokalny LLM)
- Piper (lokalny TTS, polski)
- IC Project REST API
- FastAPI + prosty panel WWW
- systemd + Caddy
- bez Dockera i bez płatnych API

> Status: v0.2 / MVP. Projekt przeznaczony do testów laboratoryjnych i dalszego dopasowania do konkretnej konfiguracji FreePBX/IC Project.

## Architektura

```text
PSTN/SIP
   |
FreePBX / Asterisk
   |
Custom Destination / dialplan
   |
AudioSocket(TCP/PCM 8 kHz)
   |
AI Voice Agent LXC
   +-- VAD
   +-- faster-whisper
   +-- Ollama
   +-- Piper
   +-- IC Project REST API
   +-- Panel WWW
```

## Założenia

Domyślnie agent działa turami:

1. Agent odtwarza powitanie.
2. Klient mówi.
3. Po wykryciu ciszy wypowiedź trafia do Whisper.
4. LLM generuje krótką odpowiedź oraz aktualizuje strukturę zgłoszenia.
5. Piper generuje odpowiedź głosową.
6. Po zebraniu wymaganych danych agent tworzy task w IC Project.

To jest prostsze i stabilniejsze niż pełny realtime/barge-in.

## Wymagania

### Proxmox

- Proxmox VE 8.4+ / 9.x
- Debian 13 LXC (domyślnie)
- 4 vCPU
- 8 GB RAM minimum
- 20–32 GB dysku
- internet podczas instalacji

### FreePBX / Asterisk

- Asterisk 18+ z modułem `app_audiosocket`
- łączność TCP z FreePBX do LXC na porcie `9019`

Sprawdzenie:

```bash
asterisk -rx "module show like audiosocket"
```

## Instalacja LXC

Projekt zawiera dwa warianty:

### A. Styl Community Scripts

Pliki:

```text
ct/freepbx-ai-icproject.sh
install/freepbx-ai-icproject-install.sh
```

Skrypt `ct/` jest wzorowany na strukturze Community Scripts i korzysta z ich `build.func`.

Uruchom na hoście Proxmox:

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/Pawel-sp9pw/freepbx-ai-icproject/main/ct/freepbx-ai-icproject.sh)"
```

Skrypt instalacyjny domyślnie pobiera projekt z tego repozytorium.

### B. Instalacja istniejącego Debian LXC

Skopiuj katalog projektu do LXC i uruchom:

```bash
cd /opt
git clone https://github.com/Pawel-sp9pw/freepbx-ai-icproject.git
cd freepbx-ai-icproject
bash scripts/install-local.sh
```

## Panel WWW

Po instalacji:

```text
http://IP_LXC:8080
```

Domyślne logowanie jest tworzone podczas instalacji. Hasło znajdziesz tylko lokalnie:

```bash
cat /etc/freepbx-ai/admin-password
```

Zmień je po pierwszym logowaniu.

## Konfiguracja FreePBX

### 1. Sprawdź moduł

```bash
asterisk -rx "module show like audiosocket"
```

### 2. Dodaj custom dialplan

W `/etc/asterisk/extensions_custom.conf`:

```ini
[ai-icproject]
exten => s,1,NoOp(AI IC Project Agent)
 same => n,Answer()
 same => n,Set(AI_UUID=${UUID()})
 same => n,AudioSocket(${AI_UUID},192.168.1.50:9019)
 same => n,Hangup()
```

Zmień `192.168.1.50` na adres LXC.

Następnie:

```bash
fwconsole reload
```

### 3. Custom Destination

FreePBX → Admin → Custom Destinations:

```text
Target: ai-icproject,s,1
Description: AI IC Project
```

Następnie podepnij ten destination do:
- numeru DID,
- IVR,
- kolejki po timeout,
- routingu poza godzinami.

## IC Project

W panelu wpisz:

- instance slug, np. `moja-firma`
- API token
- board column UUID/ID
- opcjonalnie projekt/board, jeśli używasz dodatkowego mapowania

Agent tworzy task przez:

```text
POST https://app.icproject.com/api/instance/{INSTANCE}/project/tasks
```

Nagłówek:

```text
X-Auth-Token: ...
```

Przykładowy payload:

```json
{
  "identifier": "uuid",
  "boardColumn": "uuid-kolumny",
  "name": "Problem z programem magazynowym",
  "description": "Klient zgłasza błąd 500...",
  "priority": "normal"
}
```

## Modele

Domyślne:

- Whisper: `small`
- Ollama: `qwen3:4b`
- Piper: `pl_PL-mc_speech-medium`

Model TTS można zmienić w panelu.

## WireGuard — opcjonalny tunel do FreePBX

Jeśli agent AI i FreePBX stoją w różnych lokalizacjach, LXC może działać jako klient WireGuard. `wireguard-tools` jest instalowany automatycznie.

Konfigurację można wkleić w panelu WWW. Jest zapisywana jako `/etc/wireguard/wg0.conf` z uprawnieniami `0600`, a następnie uruchamiana przez `wg-quick@wg0`. Panel nie odczytuje ponownie klucza prywatnego.

Przykład:

```ini
[Interface]
PrivateKey = <PRYWATNY_KLUCZ_LXC>
Address = 10.20.30.2/32

[Peer]
PublicKey = <KLUCZ_SERWERA_WG>
Endpoint = vpn.example.pl:51820
AllowedIPs = 192.168.10.0/24
PersistentKeepalive = 25
```

Jeżeli FreePBX ma np. `192.168.10.20`, jego sieć musi znaleźć się w `AllowedIPs`, a po stronie serwera WireGuard musi istnieć trasa zwrotna do adresu tunelowego LXC.

Diagnostyka:

```bash
wg show wg0
systemctl status wg-quick@wg0
```

## Usługi

```bash
systemctl status freepbx-ai
systemctl status ollama
systemctl status piper-ai
systemctl status caddy
```

Logi:

```bash
journalctl -u freepbx-ai -f
```

## Porty

- `8080/tcp` — panel WWW
- `9019/tcp` — AudioSocket
- `11434/tcp` — Ollama, tylko localhost
- `5000/tcp` — Piper HTTP, tylko localhost
- WireGuard: port UDP zależny od konfiguracji peera (często `51820/udp`)

## Bezpieczeństwo

- nie wystawiaj `9019`, `11434` ani `5000` do Internetu,
- ogranicz 9019 firewallem do IP FreePBX,
- trzymaj token IC Project wyłącznie po stronie serwera,
- panel WWW wystawiaj najlepiej przez HTTPS/reverse proxy/VPN,
- ustaw długie hasło administratora,
- wykonuj backup `/var/lib/freepbx-ai` i `/etc/freepbx-ai`.

## GPU

MVP działa na CPU. Dla szybszego Whisper/Ollama można później dodać:
- passthrough NVIDIA do LXC,
- CUDA,
- model Whisper `medium`/`large-v3-turbo`.

Najpierw uruchom wersję CPU i zmierz opóźnienia.

## Aktualizacja

```bash
cd /opt/freepbx-ai-icproject
git pull
/opt/freepbx-ai/.venv/bin/pip install -r requirements.txt
systemctl restart freepbx-ai
```

## Test API bez telefonu

```bash
curl http://127.0.0.1:8000/api/health
```

Test IC Project wykonasz z panelu.

## Ważne

Przed wdrożeniem produkcyjnym:
- dostosuj prompt,
- przetestuj identyfikację klienta,
- ustal politykę priorytetów,
- dodaj informację o automatycznym asystencie/nagrywaniu zgodnie z wymaganiami firmy i prawa,
- sprawdź zachowanie po błędach ICP/LLM/STT.
