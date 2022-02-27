#!/bin/bash

mkdir model-store

curl -L https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/pytorch_model.bin -o ./model-store/pytorch_model.bin
curl https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/config.json -o ./model-store/config.json
curl https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/tokenizer.json -o ./model-store/tokenizer.json
curl https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/tokenizer_config.json -o ./model-store/tokenizer_config.json
curl https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/vocab.txt -o ./model-store/vocab.txt