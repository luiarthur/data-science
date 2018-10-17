.PHONY: build clean

run: build
	./hello.out

build: hello.out

hello.out: main.cpp State.h
	g++ main.cpp -o hello.out

clean:
	rm -f *.o *.gch *.out

