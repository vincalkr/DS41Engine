#!/usr/bin/env python3
"""followup-sizes.py TAG - a 25K-token first turn (long22k.txt), then four follow-ups of roughly 100, 150, 200 and 250
new tokens with 16-token answers: the prompt-side cost of mid-size agent turns (the ones STRATA_SERVE_STEP_MAX decides
between the exact passes and the layer-major batch).  One line per turn to local/bench-v41/followup-sizes.txt."""
import json, sys, time, urllib.request, subprocess
tag = sys.argv[1]
doc = open("local/bench-v41/long22k.txt").read()
msgs = [{"role": "user", "content": doc + "\n\nRiassumi in una frase il testo qui sopra."}]
pad = "Considera anche i casi limite, le liste vuote e i valori negativi. "
for i, reps in enumerate([0, 9, 14, 19, 24]):
    if i: msgs.append({"role": "user", "content": "Aggiungi un dettaglio sul punto %d. " % i + pad * reps})
    body = {"model": "x", "stream": False, "max_tokens": 16, "temperature": 0.0, "messages": msgs,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8150/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    msgs.append({"role": "assistant", "content": r["choices"][0]["message"]["content"]})
    time.sleep(0.5)
    pl = subprocess.run("grep -E 'serve: prompt [0-9]+ tokens' local/serve-engine.log | tail -1", shell=True,
                        capture_output=True, text=True).stdout.strip().split("serve: ")[-1][:90]
    line = f"{tag} turn {i}: {pl}"
    print(line, flush=True); open("local/bench-v41/followup-sizes.txt", "a").write(line + "\n")
    time.sleep(5)
