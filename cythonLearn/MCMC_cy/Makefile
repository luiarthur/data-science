.PHONY: lint test

lint:
	# flake8
	pylint --rcfile=pylint.config *.py
	pylint --rcfile=pylint.config *.pyx

test:
	python3 test.py
