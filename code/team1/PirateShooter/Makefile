.PHONY: generate clean

all: generate

generate:
	@echo "generating Protobuf files"
	cd protobuf \
	&& python -m grpc_tools.protoc -I recorder/proto=. --python_out=../ --pyi_out=../ --grpc_python_out=../ *.proto

clean:
	@echo "cleaning up"
	rm protobuf/**/*.py
	rm protobuf/**/*.pyi
