.PHONY: test build clean

build:
	python3 setup.py build_ext --inplace

test: build
	python3 test.py

clean:
	rm -rf *.so *.c build/*
