# Testy regresyjne

Projekt zawiera testy bezpieczeństwa i logiki rozmowy oparte o standardowy moduł `unittest`.

## Uruchomienie

Na serwerze agenta:

```bash
cd /opt/freepbx-ai-icproject
./scripts/run-tests.sh
```

lub bezpośrednio:

```bash
/opt/freepbx-ai/.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

## Zakres

Testy obejmują m.in.:

- wykrywanie typowych prób prompt injection,
- blokadę ujawniania promptów/tokenów i wykonywania poleceń,
- ograniczenie LLM do pola aktualnie oczekiwanego przez maszynę stanów,
- brak możliwości nadpisania wcześniej zebranych danych przez ogólny fallback LLM,
- brak możliwości ustawienia przez LLM pól technicznych takich jak `priority`, `caller` czy `summary`,
- ścisłe potwierdzenia `tak/nie`,
- dopasowanie CallerID do słownika klientów,
- ochronę przed błędnym dopasowaniem krótkiego numeru,
- zachowanie numeru źródłowego CallerID w zgłoszeniu IC Project,
- adnotację dla zgłoszeń wymagających doprecyzowania.

Testy nie zastępują testu telefonicznego end-to-end. Nie oceniają jakości mikrofonu, kodeka, VAD, Whispera na rzeczywistym audio ani jakości głosu Piper.
