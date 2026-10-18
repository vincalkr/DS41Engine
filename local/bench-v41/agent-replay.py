#!/usr/bin/env python3
"""agent-replay.py TAG - an agent-like session against 127.0.0.1:8150: a ~25K-token first turn (local/bench-v41/long22k.txt),
then four follow-ups of ~100-600 new tokens each.  Per turn: new prompt tokens, TTFT, decode tok/s (from the server log).
Appends one line per turn to local/bench-v41/agent-replay.txt."""
import json, sys, time, urllib.request, subprocess
tag = sys.argv[1] if len(sys.argv) > 1 else "run"
doc = open("local/bench-v41/long22k.txt").read()
msgs = [{"role": "system", "content": "Sei un assistente tecnico che lavora su questo repository."},
        {"role": "user", "content": doc + "\n\nRiassumi in 5 punti il testo qui sopra."}]
follow = ["Spiega meglio il punto 2, con un esempio concreto.",
          "Scrivi una funzione Python che calcoli la media mobile di una lista, con docstring e un test.",
          "Ora riscrivi la stessa funzione usando numpy e confronta le due versioni. " + "Considera liste molto lunghe. " * 40,
          "Riassumi in una frase tutta la conversazione."]
def last(pat):
    return subprocess.run(f"grep -E '{pat}' local/serve-engine.log | tail -1", shell=True, capture_output=True, text=True).stdout.strip()
for i in range(len(follow) + 1):
    body = {"model": "x", "stream": True, "max_tokens": 400, "temperature": 0.0, "messages": msgs,
            "chat_template_kwargs": {"enable_thinking": False}, "stream_options": {"include_usage": True}}
    t0 = time.time(); t1 = None; text = []
    with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8150/v1/chat/completions", json.dumps(body).encode(),
                                {"Content-Type": "application/json"}), timeout=3600) as r:
        for raw in r:
            l = raw.decode().strip()
            if not l.startswith("data:") or l[5:].strip() == "[DONE]": continue
            j = json.loads(l[5:])
            for c in j.get("choices", []):
                if c.get("delta", {}).get("content"):
                    if t1 is None: t1 = time.time()
                    text.append(c["delta"]["content"])
    time.sleep(0.5)
    pl, dl = last("serve: prompt [0-9]+ tokens"), last("serve: [0-9]+ tokens in")
    line = f"{tag} turn {i}: TTFT {t1 - t0:5.1f}s | {pl.split('serve: ')[-1][:60]} | {dl.split('serve: ')[-1][:45]}"
    print(line, flush=True); open("local/bench-v41/agent-replay.txt", "a").write(line + "\n")
    msgs.append({"role": "assistant", "content": "".join(text)})
    if i < len(follow): msgs.append({"role": "user", "content": follow[i]})
