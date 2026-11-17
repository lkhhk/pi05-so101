"""Convert recorded LeRobot v3 episodes into isolated OpenPI v2 train/eval datasets."""
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path('/root/autodl-tmp/so101-medicine-run')
SRC = ROOT / 'source'
TASK = 'Pick up the white medicine bottle and place it inside the brown cardboard box.'

def main():
    info = json.loads((SRC / 'meta/info.json').read_text())
    if info['total_episodes'] != 50 or info['fps'] != 30:
        raise ValueError('Unexpected source dataset')
    episodes = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(SRC / 'meta/episodes/**/*.parquet'), recursive=True))]).sort_values('episode_index')
    data = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(SRC / 'data/**/*.parquet'), recursive=True))], ignore_index=True)
    if len(episodes) != 50 or len(data) != info['total_frames']:
        raise ValueError('Source count mismatch')
    names = info['features']['action']['names']
    features = {
        'front_image': {'dtype': 'video', 'shape': (224,224,3), 'names': ['height','width','channel']},
        'wrist_image': {'dtype': 'video', 'shape': (224,224,3), 'names': ['height','width','channel']},
        'state': {'dtype':'float32','shape':(6,), 'names':names},
        'actions': {'dtype':'float32','shape':(6,), 'names':names},
    }
    held_out = {5, 15, 28, 40, 49}
    failed = {11, 24, 25}
    for split, indices in [('train', sorted(set(range(50)) - held_out - failed)), ('eval', sorted(held_out))]:
        out = ROOT / 'lerobot' / 'local' / ('so101_medicine_' + split)
        if out.exists(): raise FileExistsError(out)
        ds = LeRobotDataset.create(repo_id='local/so101_medicine_'+split, root=out, fps=30, robot_type='so_follower', features=features, use_videos=True, image_writer_threads=4)
        for index in indices:
            e=episodes.loc[episodes.episode_index == index].iloc[0]
            rows=data.loc[data.episode_index == index].sort_values('frame_index')
            if len(rows)!=int(e['length']) or not np.array_equal(rows.frame_index.to_numpy(),np.arange(len(rows))): raise ValueError(f'Invalid frame indexing in episode {index}')
            captures=[]
            for camera in ('front','wrist'):
                key='videos/observation.images.'+camera+'/'
                path=SRC/'videos'/('observation.images.'+camera)/f"chunk-{int(e[key+'chunk_index']):03d}"/f"file-{int(e[key+'file_index']):03d}.mp4"
                cap=cv2.VideoCapture(str(path)); start=round(float(e[key+'from_timestamp'])*30)
                if not cap.isOpened() or not cap.set(cv2.CAP_PROP_POS_FRAMES,start): raise RuntimeError(f'Cannot seek {path} to {start}')
                captures.append(cap)
            try:
                for _,row in rows.iterrows():
                    images=[]
                    for cap in captures:
                        ok,frame=cap.read()
                        if not ok: raise RuntimeError(f'Video missing frame in episode {index}')
                        images.append(cv2.cvtColor(cv2.resize(frame,(224,224),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB))
                    state=np.asarray(row['observation.state'],dtype=np.float32);action=np.asarray(row['action'],dtype=np.float32)
                    if state.shape!=(6,) or action.shape!=(6,) or not np.isfinite(state).all() or not np.isfinite(action).all(): raise ValueError(f'Invalid joint data {index}')
                    ds.add_frame({'front_image':images[0], 'wrist_image':images[1], 'state':state, 'actions':action, 'task':TASK})
            finally:
                for cap in captures: cap.release()
            ds.save_episode()
            print(split,'source episode',index,'frames',len(rows),flush=True)
        print(split, 'complete',flush=True)

if __name__ == '__main__': main()
