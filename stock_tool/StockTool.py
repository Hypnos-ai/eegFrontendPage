# Import statements
import gym
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy, register_policy
from stable_baselines3 import PPO

from StockEnv import StockTradingEnv

class CNNLSTMFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that applies 1D CNN followed by LSTM on the input observations.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        # Calculate the input dimensions
        super(CNNLSTMFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=observation_space.shape[1], out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Assuming the input has window_size=5 and features=19 (from StockEnv)
        # After Conv and Pool layers, calculate the output size
        # This depends on the specific architecture; adjust accordingly
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        
        # Output from LSTM will be passed to a linear layer
        self.linear = nn.Sequential(
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch_size, window_size, features)
        # Permute to (batch_size, features, window_size) for Conv1d
        x = observations.permute(0, 2, 1)
        x = self.conv(x)
        
        # Permute back to (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)
        
        # LSTM expects input of shape (batch, seq, feature)
        lstm_out, _ = self.lstm(x)
        
        # Use the last output of LSTM
        lstm_out = lstm_out[:, -1, :]
        
        x = self.linear(lstm_out)
        return x

class CustomCNNLSTMPPOPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic Policy with CNN + LSTM feature extractor.
    """
    def __init__(self, *args, **kwargs):
        super(CustomCNNLSTMPPOPolicy, self).__init__(
            *args, 
            **kwargs,
            features_extractor_class=CNNLSTMFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=256)
        )
        # Override the activation function for the policy and value networks if needed
        self.mlp_extractor.policy_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.mlp_extractor.value_net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

# Register the custom policy
register_policy('CustomCNNLSTMPPOPolicy', CustomCNNLSTMPPOPolicy)

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    val = 14
    
    # Define the features used for scaling
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'SMA', 'EMA', 'RSI', 'MACD', 'Signal_Line',
                'MACD_Hist', 'Upper_BB', 'Lower_BB', 'BB_Width',
                'ATR', 'ROC', 'OBV']
    
    # Basic indicators
    df['SMA'] = df['Close'].rolling(window=val).mean()
    df['EMA'] = df['Close'].ewm(span=val, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=val).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=val).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    # Bollinger Bands
    MA = df['Close'].rolling(window=20).mean()
    STD = df['Close'].rolling(window=20).std()
    df['Upper_BB'] = MA + (STD * 2)
    df['Lower_BB'] = MA - (STD * 2)
    df['BB_Width'] = df['Upper_BB'] - df['Lower_BB']
    
    # ATR
    df['TR'] = np.maximum(
        df['High'] - df['Low'],
        np.abs(df['High'] - df['Close'].shift(1)),
        np.abs(df['Low'] - df['Close'].shift(1))
    )
    df['ATR'] = df['TR'].rolling(window=val).mean()
    
    # ROC
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Fill NaN values
    df = df.bfill()
    
    # Feature Scaling
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Calculate new indicators
    df['NewIndicator1'] = ...  # Your calculation here
    df['NewIndicator2'] = ...  # Your calculation here
    
    # Ensure to scale the new indicators
    additional_features = ['NewIndicator1', 'NewIndicator2']
    df[additional_features] = scaler.fit_transform(df[additional_features])
    
    # Include the new indicators in self.features within StockEnv
    # Ensure StockEnv.py's self.features includes these new indicators
    
    print(f"DataFrame after calculating technical indicators and scaling: {df.shape}")
    print(df.head())
    
    return df

def train_model(df):
    """Train the PPO model with the custom CNN + LSTM policy."""
    # Define environment creation function for DummyVecEnv
    env_maker = lambda: StockTradingEnv(df=df, frame_bound=(window_size, end_index), window_size=window_size)
    env = DummyVecEnv([env_maker])
    
    # Initialize the PPO model with the custom policy
    model = PPO(
        policy='CustomCNNLSTMPPOPolicy',  # Use the registered custom policy
        env=env,
        verbose=1,
        tensorboard_log="./ppo_stock_tensorboard/"
    )
    
    # Train the model
    model.learn(total_timesteps=100000)  # Increased timesteps for better learning
    
    return model

def calculate_buy_and_hold_return(df, start_index, end_index):
    """Calculate the buy-and-hold return over the specified period."""
    initial_price = df['Open'].iloc[start_index]
    final_price = df['Close'].iloc[end_index]
    return ((final_price - initial_price) / initial_price) * 100

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate the Sharpe Ratio."""
    return (np.mean(returns) - risk_free_rate) / np.std(returns) * np.sqrt(252)  # Annualized

def calculate_max_drawdown(portfolio_values):
    """Calculate the Maximum Drawdown."""
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (cumulative_max - portfolio_values) / cumulative_max
    return np.max(drawdowns) * 100  # Percentage

def evaluate_agent(model, env, df, start_index, end_index, episodes=10):
    """Evaluate the trained agent and compare with Buy-and-Hold."""
    all_rewards = []
    all_portfolio_values = []
    buy_and_hold_returns = []
    
    for episode in range(episodes):
        obs = env.reset()
        portfolio_values = [env.env_method('reset')[0]['portfolio_value']] if isinstance(env, DummyVecEnv) else [env.portfolio_value]
        done = False
        total_reward = 0
        actions = []
        
        while not done:
            if isinstance(env, DummyVecEnv):
                # If environment is vectorized
                action, _states = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                portfolio_value = info[0]['portfolio_value'] if len(info) > 0 else env.portfolio_value
            else:
                # Standard environment
                action, _states = model.predict(obs, deterministic=False)
                obs, reward, done, info = env.step(action)
                portfolio_value = info['portfolio_value']
            
            portfolio_values.append(portfolio_value)
            total_reward += reward
            actions.append(action)
        
        all_rewards.append(total_reward)
        all_portfolio_values.append(portfolio_values)
        
        # Calculate Buy-and-Hold Return for Comparison
        buy_hold_return = calculate_buy_and_hold_return(df, start_index, end_index)
        buy_and_hold_returns.append(buy_hold_return)
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Final Portfolio Value = {portfolio_value:.2f}, Buy-and-Hold Return = {buy_hold_return:.2f}%")
    
    # Calculate Metrics
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    print(f"\nMean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Calculate and Print Sharpe Ratio and Max Drawdown for Each Episode
    for i, portfolio in enumerate(all_portfolio_values):
        returns = np.diff(portfolio) / portfolio[:-1]
        sharpe_ratio = calculate_sharpe_ratio(returns)
        max_drawdown = calculate_max_drawdown(portfolio)
        print(f"Episode {i + 1} - Sharpe Ratio: {sharpe_ratio:.2f}, Max Drawdown: {max_drawdown:.2f}%")
    
    # Plot Portfolio Values
    plt.figure(figsize=(15, 6))
    for i, portfolio in enumerate(all_portfolio_values):
        plt.plot(portfolio, label=f'Episode {i+1}')
    plt.title('Agent Portfolio Value Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()
    
    # Plot Buy-and-Hold Return
    plt.figure(figsize=(15, 6))
    for i in range(episodes):
        buy_hold_portfolio = [10000 * (1 + buy_and_hold_returns[i]/100)] * len(all_portfolio_values[i])
        plt.plot(buy_hold_portfolio, label='Buy and Hold' if i == 0 else "", linestyle='--', color='black')
        plt.plot(all_portfolio_values[i], label='Agent Portfolio Value' if i == 0 else "")
    plt.title('Agent vs Buy-and-Hold Portfolio Value Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()

def main():
    # Fetch data
    df = yf.download('AMZN', period='6mo', interval='1h')  # Increased data duration
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Additional integrity checks
    if df.empty:
        raise ValueError("DataFrame is empty after calculating technical indicators.")
    if len(df) < 5:
        raise ValueError(f"DataFrame has insufficient data: {len(df)} rows.")
    
    # Define environment parameters
    window_size = 5
    start_index = window_size
    end_index = len(df) - 1  # Use the full length of available data
    
    # Create and train model
    model = train_model(df)
    
    # Evaluate model using SB3's evaluate_policy
    env = DummyVecEnv([lambda: StockTradingEnv(df=df, window_size=window_size)])
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=False)
    
    print(f"\nInitial Evaluation with evaluate_policy:")
    print(f"Mean Reward: {mean_reward} +/- {std_reward}")
    
    # Comprehensive Profit-Based Evaluation
    print("\nComprehensive Profit-Based Evaluation:")
    evaluate_agent(model, env, df, start_index, end_index, episodes=10)
    
    # Optionally, save the model
    model.save("a2c_amzn_trading_model")
    print("\nModel saved as 'a2c_amzn_trading_model'")

if __name__ == "__main__":
    main()







