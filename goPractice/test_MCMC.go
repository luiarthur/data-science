package main

import (
	"MCMC"
	"fmt"
)

type MyState struct {
	x int
	y float32
}

func (state MyState) Update() {
	state.x += 1
	state.y -= 2
}

func main() {
	state := MyState{x: 1, y: 2}

	monitors := []string{"x"}
	out := MCMC.Gibbs(state, monitors, 10, 5, 5)

	fmt.Println(out[0])
}
