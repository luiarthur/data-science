context("gibbs stuff")
library(mcmc)


# This isn't gonna work...
# Eventually, to modify state, Rlist conversions will need to take place.
update_fn <- function(state) {
  state$a <<- state$a + 1
  state$b <<- state$b - 1
}

B <- 10
burn <- 2
model <- gibbs(list(a=1, b=2), update_fn, B=B, burn=burn)


test_that("gibbs works", {
  expect_equal(model$a, B + burn)
  expect_equal(model$b, -(B + burn))
})
