cp_substate <- function(state, substate) {
  out <- NULL

  for (s in substate) {
    out$s  <- state$s
  }

  out
}

gibbs <- function(state, update_fn, B, burn, print_freq=0, monitors=NULL, thin=NULL) {
  out <- rep(list(state), B)

  for (i in 1:burn) {
    update_fn(state)
  }

  for (i in 1:B) {
    update_fn(state)
    out[[i]] <- state
  }

  return(out)
}
