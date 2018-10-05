.PHONY: test build clean

test: build
	python3 test.py

build:
	python3 setup.py build_ext --inplace

clean:
	rm -rf *.so *.c build/*
