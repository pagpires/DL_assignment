#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --cuda
elif [ "$1" = "test" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en ./outputs/test_outputs.txt --cuda
elif [ "$1" = "train_local" ]; then
	python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json
elif [ "$1" = "test_local" ]; then
    python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en ./outputs/test_outputs.txt
elif [ "$1" = "vocab" ]; then
	python vocab.py --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en vocab.json
elif [ "$1" = "dev_vocab" ]; then
	python vocab.py --train-src=./en_es_data/train_small.es --train-tgt=./en_es_data/train_small.en vocab_small.json
elif [ "$1" = "dev_train" ]; then
	python run.py train --train-src=./en_es_data/train_small.es --train-tgt=./en_es_data/train_small.en --dev-src=./en_es_data/train_small.es --dev-tgt=./en_es_data/train_small.en --vocab=vocab_small.json
elif [ "$1" = "dev_test" ]; then
	echo "Performance of untrained model"
    python run.py decode model_benchmark.bin ./en_es_data/train_small.es ./en_es_data/train_small.en ./outputs/dev_outputs.txt
	echo "Performance of trained model"
    python run.py decode model.bin ./en_es_data/train_small.es ./en_es_data/train_small.en ./outputs/dev_outputs.txt
else
	echo "Invalid Option Selected"
fi
