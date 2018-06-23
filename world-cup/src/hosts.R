library(rcommon)

"%+%" = function(a,b) paste0(a,b)

cup = read.csv('../data/hosts.csv')
N = NROW(cup)

host_won = function(host, winner) {
  if (grepl('&', host)) {
    hosts = strsplit(host, '&')
    hosts = c(sapply(hosts, trimws))
    winner %in% hosts
  } else {
    trimws(host) == trimws(winner)
  }
}


hosts = as.character(cup$host)
winners = as.character(cup$winner)

hosts_won = sapply(1:N, function(i) host_won(hosts[i], winners[i]))

prob.host.wins = mean(hosts_won)
ci.prob.host.wins = prob.host.wins + c(-1,1) * 1.96 * sd(hosts_won) / N
ci.str = "(" %+% paste(round(ci.prob.host.wins,3), collapse=', ') %+% ")"

cat('Prob. of host winning: ', prob.host.wins, ci.str, '\n')
