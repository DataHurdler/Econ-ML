library(R6)

# Define the BayesianAB class
BayesianAB <- R6Class(
  "BayesianAB",
  public = list(
    num_actions = NULL,
    num_trials = NULL,
    action_counts = NULL,
    action_rewards = NULL,
    
    initialize = function(num_actions, num_trials, action_counts, action_rewards) {
      self$num_actions <- num_actions
      self$num_trials <- num_trials
      self$action_counts <- action_counts
      self$action_rewards <- action_rewards
    },
    
    # Method for epsilon-greedy algorithm
    epsilon_greedy = function(epsilon) {
      action_values <- rep(0, self$num_actions)
      action_value_history <- matrix(0, nrow = self$num_trials, ncol = self$num_actions)
      
      for (t in 1:self$num_trials) {
        # Epsilon-greedy action selection
        if (runif(1) < epsilon) {
          action <- sample(1:self$num_actions, 1)
        } else {
          action <- which.max(action_values)
        }
        
        # Update the action counts
        self$action_counts[action] <- self$action_counts[action] + 1
        
        # Simulate the reward for the chosen action
        reward <- simulate_reward(action)
        
        # Update the action rewards
        self$action_rewards[action] <- self$action_rewards[action] + (reward - self$action_rewards[action]) / self$action_counts[action]
        
        # Update the action values
        action_values <- self$action_rewards
        
        # Store action values at current time step
        action_value_history[t, ] <- action_values
      }
      
      plot_action_values(action_value_history)
      
      return(action_values)
    },
    
    # Method for optimistic initial values algorithm
    optimistic_initial_values = function(initial_value) {
      action_values <- rep(initial_value, self$num_actions)
      action_value_history <- matrix(0, nrow = self$num_trials, ncol = self$num_actions)
      
      for (t in 1:self$num_trials) {
        # Optimistic initial values action selection
        action <- which.max(action_values)
        
        # Simulate the reward for the chosen action
        reward <- simulate_reward(action)
        
        # Update the action counts
        self$action_counts[action] <- self$action_counts[action] + 1
        
        # Update the action rewards
        self$action_rewards[action] <- self$action_rewards[action] + (reward - self$action_rewards[action]) / self$action_counts[action]
        
        # Update the action values
        action_values <- self$action_rewards
        
        # Store action values at current time step
        action_value_history[t, ] <- action_values
      }
      
      plot_action_values(action_value_history)
      
      return(action_values)
    },
    
    # Method for UCB1 algorithm
    ucb1 = function() {
      action_values <- rep(0, self$num_actions)
      action_counts <- rep(1, self$num_actions)
      action_value_history <- matrix(0, nrow = self$num_trials, ncol = self$num_actions)
      
      for (t in 1:self$num_trials) {
        # UCB1 action selection
        exploration_bonus <- sqrt(2 * log(t) / action_counts)
        action <- which.max(action_values + exploration_bonus)
        
        # Update the action counts
        self$action_counts[action] <- self$action_counts[action] + 1
        
        # Simulate the reward for the chosen action
        reward <- simulate_reward(action)
        
        # Update the action rewards
        self$action_rewards[action] <- self$action_rewards[action] + (reward - self$action_rewards[action]) / self$action_counts[action]
        
        # Update the action values
        action_values <- self$action_rewards
        
        # Store action values at current time step
        action_value_history[t, ] <- action_values
      }
      
      plot_action_values(action_value_history)
      
      return(action_values)
    },
    
    # Method for gradient bandits algorithm
    gradient_bandits = function(step_size) {
      action_preferences <- rep(0, self$num_actions)
      action_counts <- rep(0, self$num_actions)
      action_value_history <- matrix(0, nrow = self$num_trials, ncol = self$num_actions)
      
      for (t in 1:self$num_trials) {
        # Softmax action selection
        action_probabilities <- exp(action_preferences) / sum(exp(action_preferences))
        action <- sample(1:self$num_actions, 1, prob = action_probabilities)
        
        # Update the action counts
        action_counts[action] <- action_counts[action] + 1
        
        # Simulate the reward for the chosen action
        reward <- simulate_reward(action)
        
        # Update the action preferences
        baseline <- mean(self$action_rewards)
        one_hot_encoding <- ifelse(seq_len(self$num_actions) == action, 1, 0)
        action_preferences <- action_preferences + step_size * (reward - baseline) * (one_hot_encoding - action_probabilities)
        
        # Update the action values
        action_values <- exp(action_preferences) / sum(exp(action_preferences))
        
        # Store action values at current time step
        action_value_history[t, ] <- action_values
      }
      
      plot_action_values(action_value_history)
      
      return(action_values)
    },
    
    # Method for Thompson sampling algorithm
    thompson_sampling = function() {
      action_successes <- rep(0, self$num_actions)
      action_failures <- rep(0, self$num_actions)
      action_value_history <- matrix(0, nrow = self$num_trials, ncol = self$num_actions)
      
      for (t in 1:self$num_trials) {
        # Thompson sampling action selection
        action_samples <- rbeta(self$num_actions, action_successes + 1, action_failures + 1)
        action <- which.max(action_samples)
        
        # Update the action counts
        self$action_counts[action] <- self$action_counts[action] + 1
        
        # Simulate the reward for the chosen action
        reward <- simulate_reward(action)
        
        # Update the action successes and failures
        if (reward == 1) {
          action_successes[action] <- action_successes[action] + 1
        } else {
          action_failures[action] <- action_failures[action] + 1
        }
        
        # Update the action values
        action_values <- action_successes / (action_successes + action_failures)
        
        # Store action values at current time step
        action_value_history[t, ] <- action_values
      }
      
      plot_action_values(action_value_history)
      
      return(action_values)
    }
  ),
  
  private = list(
    # Function to simulate reward for a given action
    simulate_reward = function(action) {
      # Simulate reward based on the true rewards
      reward <- rbinom(1, 1, self$prob_true[action])
      
      return(reward)
    },
    
    # Function to plot action values over time
    plot_action_values = function(action_value_history) {
      # Plot action values
      plot(action_value_history, type = "l", xlab = "Time Step", ylab = "Action Value",
           main = "Action Values over Time", col = rainbow(self$num_actions))
      legend("topright", legend = paste0("Action ", 1:self$num_actions), col = rainbow(self$num_actions), lty = 1)
    }
  )
)

# Set the parameters
num_actions <- 5
num_trials <- 1000
action_counts <- rep(0, num_actions)
action_rewards <- rep(0, num_actions)

# Create an instance of the BayesianAB class
bandit <- BayesianAB$new(num_actions, num_trials, action_counts, action_rewards)

# Run the algorithms
bandit$epsilon_greedy(0.1)
bandit$optimistic_initial_values(5)
bandit$ucb1()
bandit$gradient_bandits(0.1)
bandit$thompson_sampling()
