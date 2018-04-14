# ##################################################################################### #
# NAME: BANDITS.R
# DESCRIPTION: script to implement UCB algorithm for the multi-armed bandit problem.
# ##################################################################################### #
## BEGIN SCRIPT ----------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------- #
library(tidyverse)
set.seed(3)

# Kullback Liebler function for Bernoulli distributions
kl_bernoulli <- function(p,q){
      p*log(p/q) + (1-p)*log((1-p)/(1-q))
}

# algorithm parameters
N_ROUNDS <- 2000
N_ARMS <- 3
ALPHA <- 2.2

# bandit parameters
mu_arms <- runif(N_ARMS, 0.4, 0.8)
mu_best <- max(mu_arms)
sub_opt <- mu_best - mu_arms
kl_div <- kl_bernoulli(mu_best, mu_arms)

# initialize per-round storage vectors
mu_hat <- rep(0, N_ARMS)
n_pulls <- rep(1, N_ARMS)
rewards <- rep(0, N_ARMS)

# bookkeeping
mu_hat_rounds <- matrix(rep(0, N_ARMS * N_ROUNDS), nrow = N_ROUNDS)
ucbs_rounds <- matrix(rep(0, N_ARMS * N_ROUNDS), nrow = N_ROUNDS)
arm_rounds <- rep(0, N_ROUNDS)
regret_pseudo <- rep(0, N_ROUNDS)
regret_bound <- rep(0, N_ROUNDS)
regret_bound_lo <- rep(0, N_ROUNDS)

# play N_ARMS exploration rounds
for(j in 1:N_ARMS){
      rewards[j] <- rbinom(1, 1, prob = mu_arms[j])
}
mu_hat <- rewards / n_pulls

# play N_ARMS + N_ROUNDS rounds
for(i in 1:N_ROUNDS){
      # select arm according to UCB
      ucbs <- mu_hat + sqrt(ALPHA * 0.5 * log(i+N_ARMS) / n_pulls)
      arm_idx <- which(ucbs == max(ucbs))
      if(length(arm_idx) > 1){
            idx <- sample(length(arm_idx), 1)
            arm_idx <- arm_idx[idx]
      }
      # pull selected arm
      rewards[arm_idx] <- rewards[arm_idx] + rbinom(1, 1, mu_arms[arm_idx])
      n_pulls[arm_idx] <- n_pulls[arm_idx] + 1
      # update mean estimates
      mu_hat[arm_idx] <- rewards[arm_idx] / n_pulls[arm_idx]
      # bookkeeping
      regret_pseudo[i] <- sum(sub_opt * n_pulls)
      regret_bound[i] <- sum(ALPHA*sub_opt[sub_opt > 0]/(2*sub_opt[sub_opt > 0]^2)*
                                   log(i+N_ARMS) + ALPHA / (ALPHA - 2))
      regret_bound_lo[i] <- sum(sub_opt[sub_opt > 0]*log(i+N_ARMS)/kl_div[kl_div > 0])
      ucbs_rounds[i,] <- ucbs
      mu_hat_rounds[i,] <- mu_hat
      arm_rounds[i] <- arm_idx
}

ggplot() + geom_point(aes(x = 1:N_ROUNDS, y = as.factor(arm_rounds))) + 
      labs(x = "rounds", y = "arms")

ggplot() + geom_bar(aes(as.factor(arm_rounds))) + labs(x = "rounds", y = "arms")

ggplot() + geom_step(aes(x = 1:N_ROUNDS, y = mu_hat_rounds[,1])) +
      geom_step(aes(x = 1:N_ROUNDS, y = mu_hat_rounds[,2]), color = "red") +
      geom_step(aes(x = 1:N_ROUNDS, y = mu_hat_rounds[,3]), color = "blue") + 
      labs(x = "rounds", y = "mean estimates")

ggplot() + geom_line(aes(x = 1:N_ROUNDS, y = regret_pseudo), color = "green4") +
      geom_step(aes(x = 1:N_ROUNDS, y = regret_bound), color = "red")  +
      geom_step(aes(x = 1:N_ROUNDS, y = regret_bound_lo), color = "blue")  + 
      labs(x = "rounds", y = "regret")

# ggplot() + geom_step(aes(x = 1:N_ROUNDS, y = ucbs_rounds[,1])) +
#       geom_step(aes(x = 1:N_ROUNDS, y = ucbs_rounds[,2]), color = "red")  +
#       geom_step(aes(x = 1:N_ROUNDS, y = ucbs_rounds[,3]), color = "blue") + 
#       labs(x = "rounds", y = "UCB estimates")

# ggplot() + geom_step(aes(x=1:N_ROUNDS, y = mu_arms[1] - mu_hat_rounds[,1])) +
#       geom_step(aes(x=1:N_ROUNDS, y = mu_arms[2] - mu_hat_rounds[,2]), color="red") +
#       geom_step(aes(x=1:N_ROUNDS, y = mu_arms[3] - mu_hat_rounds[,3]), color="blue") + 
#       labs(x = "rounds", y = "mean estimates deviations")

# ggplot() + geom_step(aes(x = 1:N_ROUNDS, y = arm_rounds))+ 
#       labs(x = "rounds", y = "arms")
# ##################################################################################### #
## END OF SCRIPT ---------------------------------------------------------------------- #
# ##################################################################################### #