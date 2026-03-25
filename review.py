def cohens_kappa(items, labels):
    """Compute Kohen's Kappa for 2-rater agreement on `labels`.  Generally, anything above .8 is considered good"""
    mapping = {l[0]:i for i, l in enumerate(labels)}
    n_k1 = [0 for _ in labels]
    n_k2 = [0 for _ in labels]
    n_o = 0
    n = 0
    for item in items:
        n+=1
        if item['result']:
            n_o+=1
        n_k1[mapping[item['rater1']['result']]] += 1
        n_k2[mapping[item['rater2']['result']]] += 1
    p_o = n_o/n
    p_e = sum(x[0]*x[1] for x in zip(n_k1, n_k2))/(n**2)
    print(mapping, n_k1, n_k2, p_o, p_e)
    return (p_o-p_e)/(1-p_e)

def aggregate_disagreements(items, n_examples = 2, order_independent = True):
    disagreements = {}
    for item in items:
        if not item['result']:
            key = [item['rater1']['result'], item['rater2']['result']]
            if order_independent:
                key = sorted(key)
            key = ', '.join(key)
            if key not in disagreements:
                disagreements[key] = []
            if len(disagreements[key])<n_examples:
                disagreements[key].append(item['parent'])
    return disagreements
