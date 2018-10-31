.PHONY: lint test build clean

lint:
	# flake8
	pylint --rcfile=pylint.config *.py
	pylint --rcfile=pylint.config *.pyx

build:
	python3 setup.py build_ext --inplace

test:
	python3 test.py

clean:
	rm -rf *.c *.so build
