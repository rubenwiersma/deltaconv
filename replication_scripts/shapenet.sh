#!/bin/bash
python experiments/train_shapenet.py --checkpoint_dir experiments/runs/shapenet/replication/checkpoint
python experiments/test_shapenet.py --checkpoint experiments/runs/shapenet/replication/checkpoint/last.pt