#!/usr/bin/env python3
"""think-budget.py [budgets...] - thinking-mode checks against 127.0.0.1:8150 at several reasoning_budget_tokens: per
(prompt, budget) the finish reason, reasoning/answer sizes, whether the budget cut the thinking, and three corruption
checks on the answer (think tags, 8-word passages repeated >= 3 times, non-Latin characters).  Output also saved as JSON."""
import json, re, sys, time, urllib.request, collections
budgets = [int(x) for x in sys.argv[1:]] or [192, 512, 1024, 2048]
prompts = [
    ("divina", "descrivimi in breve la Divina Commedia"),
    ("logica", "Anna è più alta di Bruno. Carlo è più basso di Bruno ma più alto di Dario. Elena è più alta di Anna. "
               "Ordina le cinque persone dalla più alta alla più bassa e spiega il ragionamento."),
    ("codice", "Scrivi una funzione Python che, data una lista di intervalli [inizio, fine], li unisca quando si "
               "sovrappongono e restituisca la lista ordinata. Aggiungi due esempi d'uso."),
    ("fatti", "Chi ha scritto I Promessi Sposi, in che anno è uscita l'edizione definitiva e in quale città è ambientata "
              "la prima parte del romanzo?"),
]
def rep8(text):
    w = re.findall(r"\w+", text.lower()); c = collections.Counter(tuple(w[i:i+8]) for i in range(len(w) - 7))
    return sum(1 for v in c.values() if v >= 3)
out = []
for b in budgets:
    for tag, q in prompts:
        body = {"model": "x", "stream": False, "max_tokens": b + 1500, "temperature": 0.8, "reasoning_budget_tokens": b,
                "messages": [{"role": "user", "content": q}], "chat_template_kwargs": {"enable_thinking": True}}
        t0 = time.time()
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8150/v1/chat/completions",
                         json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
        except Exception as e:
            print(f"budget {b:5d} {tag:7s} ERROR {e}"); out.append({"budget": b, "tag": tag, "error": str(e)}); continue
        m = r["choices"][0]["message"]; ans = (m.get("content") or ""); rea = (m.get("reasoning_content") or "")
        cut = "thought about this long enough" in rea
        bad = []
        if re.search(r"</?think>", ans): bad.append("think-tag")
        if rep8(ans): bad.append(f"loop{rep8(ans)}")
        if re.search(r"[Ѐ-ӿ一-鿿぀-ヿ]", ans): bad.append("foreign")
        if not ans.strip(): bad.append("EMPTY")
        dt = time.time() - t0; ct = r["usage"]["completion_tokens"]
        print(f"budget {b:5d} {tag:7s} finish={r['choices'][0]['finish_reason']:6s} tokens={ct:5d} {dt:6.1f}s "
              f"({ct/dt:5.2f} tok/s incl. prompt) reasoning={len(rea):5d}ch cut={'yes' if cut else 'no ':3s} "
              f"answer={len(ans):5d}ch {'OK' if not bad else 'BAD ' + ','.join(bad)}")
        out.append({"budget": b, "tag": tag, "finish": r["choices"][0]["finish_reason"], "tokens": ct, "secs": dt,
                    "cut": cut, "bad": bad, "reasoning": rea, "answer": ans})
json.dump(out, open("local/bench-v41/think-budget.json", "w"), ensure_ascii=False, indent=1)
