#!/usr/bin/env python3
"""code-corrupt.py N - the interval-merge code prompt, thinking on (budget 1024), temperature 0.8, N times with the chat's
repetition_penalty 1.1 and N times with 1.0 (alternated): how many answers' first code block does not run / fails the tests."""
import json, re, subprocess, sys, time, urllib.request
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
Q = ("Scrivi una funzione Python che, data una lista di intervalli [inizio, fine], li unisca quando si sovrappongono e "
     "restituisca la lista ordinata. Aggiungi due esempi d'uso.")
TESTS = r'''
import inspect
f=[v for k,v in list(globals().items()) if inspect.isfunction(v)][0]
cases=[([[1,3],[2,6],[8,10],[15,18]],[[1,6],[8,10],[15,18]]),([[1,4],[4,5]],[[1,5]]),([[5,7],[1,2],[2,3]],[[1,3],[5,7]]),([],[]),([[1,10],[2,3],[4,5]],[[1,10]])]
print(sum([list(map(list,f([list(x) for x in a])))==e for a,e in cases]), "/", len(cases))
'''
res = []
for i in range(N):
    for rp in (1.1, 1.0):
        body = {"model": "x", "stream": False, "max_tokens": 2500, "temperature": 0.8, "top_p": 0.95, "repetition_penalty": rp,
                "reasoning_budget_tokens": 1024, "messages": [{"role": "user", "content": Q}], "chat_template_kwargs": {"enable_thinking": True}}
        t0 = time.time()
        r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8150/v1/chat/completions",
                     json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600))
        ans = r["choices"][0]["message"].get("content") or ""
        blocks = re.findall(r"```(?:python)?\n(.*?)```", ans, re.S)
        if not blocks:
            verdict = "NO CODE"
        else:
            open("/tmp/claude-1000/codetest/cc.py", "w").write(blocks[0] + "\n" + TESTS)
            p = subprocess.run(["python3", "-I", "/tmp/claude-1000/codetest/cc.py"], capture_output=True, text=True, timeout=20)
            out = (p.stdout + p.stderr).strip().splitlines()
            verdict = out[-1] if out else "?"
        print(f"rep={rp} run {i}: {verdict}  ({time.time()-t0:.0f}s, {r['usage']['completion_tokens']} tok)", flush=True)
        res.append({"rep": rp, "run": i, "verdict": verdict, "answer": ans})
json.dump(res, open("local/bench-v41/code-corrupt.json", "w"), ensure_ascii=False, indent=1)
