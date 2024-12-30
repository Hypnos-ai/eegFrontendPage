import os
import pickle

class SessionConfig:
    def __init__(self):
        self.actions = {}          # Store action configurations
        self.repetitions = 21
        self.break_duration = 3
        self.record_duration = 5
        self.lowcut = 0.5
        self.highcut = 45.0
        self.channel_configs = {}  # Add this line to store channel configurations
        self.base_path = os.getenv('NEUROSYNC_PATH', 'C:\\NeuroSync')

    def to_dict(self):
        """
        Convert this SessionConfig object into a plain dictionary
        which can be pickled safely, even under Cython.
        """
        return {
            "actions": self.actions,
            "repetitions": self.repetitions,
            "break_duration": self.break_duration,
            "record_duration": self.record_duration,
            "lowcut": self.lowcut,
            "highcut": self.highcut,
            "channel_configs": self.channel_configs  # Add this line to include channel configs
        }

    @classmethod
    def from_dict(cls, data):
        """
        Reconstruct a SessionConfig object from a dictionary
        """
        obj = cls()
        obj.actions = data.get("actions", {})
        obj.repetitions = data.get("repetitions", 21)
        obj.break_duration = data.get("break_duration", 3)
        obj.record_duration = data.get("record_duration", 5)
        obj.lowcut = data.get("lowcut", 0.5)
        obj.highcut = data.get("highcut", 45.0)
        obj.channel_configs = data.get("channel_configs", {})  # Add this line to load channel configs
        return obj

    def save(self, session_num):
        """Save configuration to file (as a dict)."""
        config_dir = os.path.join(self.base_path, 'data', 'configs')
        os.makedirs(config_dir, exist_ok=True)
        
        filename = os.path.join(config_dir, f'session{session_num}_config.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(self.to_dict(), f)
    
    @staticmethod
    def load(session_num, base_path=os.getenv('NEUROSYNC_PATH', 'C:\\NeuroSync')):
        """Load configuration from file and rebuild the SessionConfig object."""
        config_dir = os.path.join(base_path, 'data', 'configs')
        filename = os.path.join(config_dir, f'session{session_num}_config.pkl')
        
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                return SessionConfig.from_dict(data)
        return None

    @staticmethod
    def load_last_config(base_path=os.getenv('NEUROSYNC_PATH', 'C:\\NeuroSync')):
        """Load most recent configuration."""
        config_dir = os.path.join(base_path, 'data', 'configs')
        if not os.path.exists(config_dir):
            return None

        config_files = sorted(
            f for f in os.listdir(config_dir) if f.endswith('_config.pkl')
        )
        if not config_files:
            return None

        last_file = os.path.join(config_dir, config_files[-1])
        with open(last_file, 'rb') as f:
            data = pickle.load(f)
            return SessionConfig.from_dict(data)
