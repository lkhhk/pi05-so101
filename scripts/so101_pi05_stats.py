"""Compute SO-101 normalization from the selected training episodes only."""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from openpi.shared import normalize

root=Path('/root/autodl-tmp/so101-medicine-run')
out=root/'assets/so101_medicine'
if out.exists(): raise FileExistsError(out)
data=pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(root/'source/data/**/*.parquet'),recursive=True))],ignore_index=True)
held_out={5,15,28,40,49}
failed={11,24,25}
selected=set(range(50))-held_out-failed
train=data[data.episode_index.isin(selected)]
if set(train.episode_index.unique())!=selected: raise ValueError('Training episode set incomplete')
stats={}
for name,key in [('state','observation.state'),('actions','action')]:
    values=np.stack(train[key]).astype(np.float32)
    if values.shape[1:]!=(6,) or not np.isfinite(values).all(): raise ValueError(name)
    running=normalize.RunningStats();running.update(values);stats[name]=running.get_statistics()
normalize.save(out,stats)
print('training episodes',train.episode_index.nunique(),'frames',len(train),'statistics',out)
