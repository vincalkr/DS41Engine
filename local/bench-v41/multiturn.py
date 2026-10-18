#!/usr/bin/env python3
"""multiturn.py FILE [--think] - an agent-like conversation against 127.0.0.1:8150: turn 1 = system + FILE + a question,
turn 2 = the same messages + turn 1's answer + a new question.  Prints each turn's prompt tokens, TTFT and the server's
reuse line from local/serve-engine.log, so a broken prefix reuse shows as 'reused=0' on turn 2."""
import json, sys, time, urllib.request, subprocess
think = "--think" in sys.argv
doc = open(sys.argv[1]).read()
msgs = [{"role": "system", "content": "Sei un assistente tecnico. Rispondi in italiano, in modo conciso."},
        {"role": "user", "content": doc + "\n\nRiassumi in 3 punti il testo qui sopra."}]
def ask(msgs):
    body = {"model": "x", "stream": False, "max_tokens": 300, "temperature": 0.0, "messages": msgs,
            "chat_template_kwargs": {"enable_thinking": think}}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8150/v1/chat/completions",
                 json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
    m = r["choices"][0]["message"]
    reuse = subprocess.run("grep 'serve: reuse' local/serve-engine.log | tail -1", shell=True, capture_output=True, text=True).stdout.strip()
    print(f"prompt_tokens={r['usage']['prompt_tokens']} total={time.time()-t0:.1f}s  | {reuse}")
    return m
m1 = ask(msgs)
msgs.append({"role": "assistant", "content": m1.get("content") or ""})
msgs.append({"role": "user", "content": "Ora spiega meglio il secondo punto."})
m2 = ask(msgs)
msgs.append({"role": "assistant", "content": m2.get("content") or ""})
msgs.append({"role": "user", "content": "Grazie. Ultima domanda: qual è il punto più importante?"})
ask(msgs)
