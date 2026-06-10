import pickle, numpy as np
d = pickle.load(open('E:/ClaudeCodeWorkplace/2026-5-12-参数优化/model/output/adversarial_augmented.pkl','rb'))
for k,v in d.items():
    if isinstance(v, np.ndarray):
        print(f'{k}: shape={v.shape}, dtype={v.dtype}')
    elif isinstance(v, list):
        print(f'{k}: list len={len(v)}, first_type={type(v[0]).__name__}')
    else:
        print(f'{k}: {type(v).__name__} = {v}')
