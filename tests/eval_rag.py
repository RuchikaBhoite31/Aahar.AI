"""python -m tests.eval_rag  -> retrieval Hit@1 / Hit@3 / Hit@5, MRR, and abstention accuracy."""
import sys
from aahar.pipeline import retriever, ask
from aahar.config import settings
from tests.gold import GOLD, OOS, IN_SCOPE


def run(verbose=True):
    r = retriever()
    h1 = h3 = h5 = 0; rr = 0.0; misses = []
    for q, gold in GOLD:
        hits = r.search(q, k=5)
        rank = next((i for i, h in enumerate(hits, 1) if any(g.lower() in h.chunk.title.lower() for g in gold)), None)
        if rank: rr += 1 / rank; h1 += rank == 1; h3 += rank <= 3; h5 += 1
        else: misses.append((q, [h.chunk.title for h in hits[:3]]))
        if verbose and (not rank or rank > 1):
            print(f"  rank={rank}  {q}\n      got: {[h.chunk.title[-48:] for h in hits[:3]]}")
    # robustness: same questions lowercased with punctuation stripped must rank the same way
    import re
    rob = 0
    for q, gold in GOLD:
        q2 = re.sub(r"[^a-z0-9 ]", "", q.lower())
        top = r.search(q2, k=1)
        rob += bool(top and any(g.lower() in top[0].chunk.title.lower() for g in gold))
    n = len(GOLD)
    oos_ok = sum(ask(q)["mode"] == "abstain" for q in OOS)
    ins_ok = sum(ask(q)["mode"] != "abstain" for q in IN_SCOPE)
    res = dict(n=n, hit1=h1 / n, hit3=h3 / n, hit5=h5 / n, mrr=rr / n, hit1_normalised_variant=rob / n, oos_refused=oos_ok / len(OOS), in_scope_answered=ins_ok / len(IN_SCOPE))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})
    return res


if __name__ == "__main__":
    run()
