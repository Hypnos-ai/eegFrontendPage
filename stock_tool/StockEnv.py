import gym
import numpy as np
from gym import spaces

class StockTradingEnv(gym.Env):
    def __init__(self, df, window_size=5):
        super(StockTradingEnv, self).__init__()
        
        self.df = df
        self.window_size = window_size
        self.current_step = window_size
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        
        # Include any additional technical indicators here
        self.features = [
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'SMA', 'EMA', 'RSI', 'MACD', 'Signal_Line',
            'MACD_Hist', 'Upper_BB', 'Lower_BB', 'BB_Width',
            'ATR', 'ROC', 'OBV',
            'NewIndicator1',
            'NewIndicator2',
        ]
        
        # Extend observation space to include balance and portfolio value
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size, len(self.features) + 2),  # Added 2 additional features
            dtype=np.float32
        )
        
        self.transaction_cost = 0.001  # 0.1% transaction cost
        
        # Initialize portfolio value
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        
        self.reset()
        
    def reset(self):
        self.current_step = self.window_size
        self.total_profit = 0
        self.position = 0  # 0: Neutral, 1: Long
        self.buy_price = 0
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        obs = self._next_observation()
        return obs
        
    def _next_observation(self):
        # Get the data points for the current window
        start = self.current_step - self.window_size
        end = self.current_step
        window_data = self.df[self.features].iloc[start:end].values
        
        if window_data.shape[0] != self.window_size:
            window_data = np.zeros((self.window_size, len(self.features)), dtype=np.float32)
        else:
            window_data = window_data.astype(np.float32)
        
        # Append balance and portfolio value to the observation
        extended_observation = np.append(window_data, 
                                        [[self.balance, self.portfolio_value]] * self.window_size, 
                                        axis=1)
        
        return extended_observation
    
    def step(self, action):
        current_price = self.df['Close'].iloc[self.current_step -1].item()
        next_price = self.df['Close'].iloc[self.current_step].item()
        
        old_portfolio_value = self.portfolio_value  # Store old portfolio value before action
        reward = 0  # Initialize reward
        
        if action == 1 and self.position == 0:  # Buy
            self.buy_price = next_price
            self.position = 1
            self.balance -= next_price * self.transaction_cost  # Deduct transaction cost
            reward = 0  # No immediate profit
        elif action == 2 and self.position == 1:  # Sell
            profit = (next_price - self.buy_price) / self.buy_price
            self.total_profit += profit
            self.balance += next_price * (1 - self.transaction_cost)
            self.position = 0
            reward = (self.portfolio_value + (next_price - self.buy_price)) - old_portfolio_value - next_price * self.transaction_cost
        else:  # Hold or invalid action
            reward = 0  # No change in profit
        
        self.current_step += 1
        
        done = self.current_step >= len(self.df) - 1
        
        # Update portfolio value
        self.portfolio_value = self.balance + (self.position * next_price)
        
        # Calculate reward as change in portfolio value
        reward = self.portfolio_value - old_portfolio_value
        
        obs = self._next_observation()
        
        info = {
            'total_profit': float(self.total_profit),
            'position': int(self.position),
            'portfolio_value': float(self.portfolio_value)
        }
        
        return obs, reward, done, info