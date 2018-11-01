cp_substate <- function(state, substate) {
  out <- NULL

  for (s in substates) {
    out$s  <- state$s
  }

  out
}

gibbs <- function(init, update_fn, B, burn, print_freq=0, monitors=NULL, thin=NULL) {
  out <- rep(list(init), B)

  for (i in 1:(B + burn)) {
  }
}
