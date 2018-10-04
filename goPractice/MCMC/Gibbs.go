package MCMC

import (
	//"gonum.org/v1/gonum/stat/distuv"
	//b := distuv.Bernoulli{P:.6}
	"fmt"
	"github.com/fatih/structs" // reflection
	"github.com/jinzhu/copier" // make deep copies
)

func Gibbs(init State, monitors []string, nmcmc int, nburn int,
	printFreq int) []map[string]interface{} {

	out := make([]map[string]interface{}, nmcmc)

	var state State
	copier.Copy(&state, &init)

	// Burn in
	for i := 0; i < nburn; i++ {
		state.Update()
		if printFreq > 0 && i%printFreq == 0 {
			fmt.Sprintf("%d / %d", i, nburn+nmcmc)
		}
	}

	// Collect
	for i := 0; i < nmcmc; i++ {
		state.Update()
		state_map := structs.New(state)

		for _, field := range monitors {
			out[i][field] = state_map.Field(field)
		}

		if printFreq > 0 && i%printFreq == 0 {
			fmt.Sprintf("%d / %d", i+nburn, nburn+nmcmc)
		}
	}

	return out
}
