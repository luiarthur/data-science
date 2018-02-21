my.kmeans = function(X, K, iter.max=100, init=NULL, eps=1E-5) {
  N = NROW(X)
  P = NCOL(X)

  if (is.null(init)) {
    init = X[sample(1:N, K),]
  }

  dist_x = function(x1, x2) {
    sqrt(sum((x1 - x2)^2))
  }

  curr = init
  prev = init + 1E3

  compute_centers = function(labels) {
    t(sapply(1:K, function(k) colMeans(X[which(labels==k),])))
  }

  compute_cluster = function(centers) {
    dists = apply(centers, 1, function(center) {
      apply(X, 1, dist_x, center)
    })
    apply(dists, 1, which.min)
  }

  iter = 0
  while (dist_x(curr, prev) > eps && iter < iter.max) {
    prev = curr
    labels = compute_cluster(prev)
    curr = compute_centers(labels)
    iter = iter + 1
  }

  list(labels=labels, centers=curr, iter=iter)
}
