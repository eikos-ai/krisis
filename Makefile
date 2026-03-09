MODEL_DIR := models/all-MiniLM-L6-v2
HF_BASE   := https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main

.PHONY: build run test download-model clean

build:
	go build -o krisis ./cmd/krisis

run:
	go run ./cmd/...

test:
	go test ./...

download-model: $(MODEL_DIR)/model.onnx $(MODEL_DIR)/tokenizer.json

$(MODEL_DIR)/model.onnx:
	mkdir -p $(MODEL_DIR)
	curl -L -o $@ $(HF_BASE)/onnx/model.onnx

$(MODEL_DIR)/tokenizer.json:
	mkdir -p $(MODEL_DIR)
	curl -L -o $@ $(HF_BASE)/tokenizer.json

clean:
	rm -rf models/
	go clean ./...
