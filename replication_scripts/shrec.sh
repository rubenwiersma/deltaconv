for i in {1 .. 10}
do
    rm -r experiments/data/shrec/processed
    python experiments/train_shrec.py
done