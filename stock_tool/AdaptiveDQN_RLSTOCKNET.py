import joblib
import numpy as np
import mne
import gym
import pywt
import scipy.signal
from scipy import stats
from keras.models import Sequential
from keras.layers import Dense, PReLU, Conv1D, Dropout, SpatialDropout1D, MaxPooling1D
from keras.layers import GlobalMaxPooling1D, Layer, AveragePooling1D, LSTM, Reshape, BatchNormalization
from keras.regularizers import l1_l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
import random

class AdaptiveDQNRLSTOCKNET:
    def __init__(self):
        self.ncomp = 4
        self.GLOBAL_SHAPE_LENGTH = None
        self.raw = None
        self.events = None
        self.csp_filter_objects = {}
        self.csp_transformed_data = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.num_outputs = 2
        self.scaler = StandardScaler()
        
    def _mean(self, x):
        return np.mean(x, axis=-1).reshape(-1, 1)

    def _stddev(self, x):
        return np.std(x, axis=-1).reshape(-1, 1)

    def _peaktopeak(self, x):
        return np.ptp(x, axis=-1).reshape(-1, 1)

    def _variance(self, x):
        return np.var(x, axis=-1).reshape(-1, 1)

    def _rms(self, x):
        return np.sqrt(np.mean(x**2, axis=-1)).reshape(-1, 1)

    def _abs_diff_signal(self, x):
        return np.sum(np.abs(np.diff(x, axis=-1)), axis=-1).reshape(-1, 1)

    def _skewness(self, x):
        return stats.skew(x, axis=-1).reshape(-1, 1)

    def _kurtosis(self, x):
        return stats.kurtosis(x, axis=-1).reshape(-1, 1)

    def _concat_features(self, x):
        features = np.concatenate(
            (
                self._peaktopeak(x),
                self._rms(x),
                self._abs_diff_signal(x),
                self._skewness(x),
                self._kurtosis(x),
                self._variance(x),
                self._mean(x),
                self._stddev(x)
            ),
            axis=1
        )
        return features

    def _apply_cwt(self, data, scales, wavelet_name='morl'):
        cwt_coeffs = np.array([pywt.cwt(data[i, :], scales, wavelet_name)[0] 
                              for i in range(data.shape[0])])
        return cwt_coeffs

    def featuresarray_load(self, data_array):
        features = []
        fs = 250
        for d in data_array:
            alpha = mne.filter.filter_data(d, sfreq=fs, l_freq=8, h_freq=12, verbose=False)
            beta = mne.filter.filter_data(d, sfreq=fs, l_freq=12, h_freq=30, verbose=False)
            
            alph_ftrs = self._concat_features(alpha)
            beta_ftrs = self._concat_features(beta)
            
            _, p = scipy.signal.welch(beta, fs=fs, average='median', nfft=512)
            _, p2 = scipy.signal.welch(alpha, fs=fs, average='median', nfft=512)
            
            res = np.mean([alph_ftrs, beta_ftrs], axis=0)
            res = np.concatenate((res, p, p2), axis=1)
            features.append(res)
        return np.array(features)

    def setup_eeg(self, path='formatted_data/S03.fif'):
        self.raw = mne.io.read_raw_fif(path, preload=True)
        
        ica = mne.preprocessing.ICA(n_components=len(self.raw.info['ch_names']), 
                                  random_state=42, max_iter=1000)
        ica.fit(self.raw)
        ica.apply(self.raw)
        
        montage = mne.channels.make_standard_montage('standard_1020')
        self.raw.set_montage(montage)
        
        self.events = mne.events_from_annotations(self.raw)
        return self.raw, self.events

    def process_epochs(self, event_ids=[1, 2, 3, 4, 5]):
        all_epochs = mne.Epochs(self.raw, self.events[0], event_id=event_ids,
                              tmin=-0.5, tmax=5.5, baseline=(-0.5, 1), preload=True)
        all_epochs.pick_types(meg=False, eeg=True)
        
        X_all = all_epochs.get_data()
        event_ids_all = all_epochs.events[:, -1]
        
        self.ncomp = min(4, X_all.shape[1])
        print(f"Using {self.ncomp} components based on available channels")
        
        for event_id in event_ids:
            y = (event_ids_all == event_id).astype(int)
            if np.unique(y).size < 2:
                print(f"Skipping event_id {event_id}.")
                continue

            csp = CSP(n_components=self.ncomp, norm_trace=False, transform_into='csp_space')
            csp.fit(X_all, y)
            self.csp_transformed_data[event_id] = csp.transform(X_all)
            self.csp_filter_objects[event_id] = csp

        joblib.dump(self.csp_filter_objects, 'csp_filters_ovr.pkl')
        print("CSP filters saved successfully.")
        n_trials = len(X_all)
        n_time_points = self.csp_transformed_data[1].shape[2]
        combined_features = np.zeros((n_trials, self.ncomp, n_time_points))

        for i, label in enumerate(event_ids_all):
            csp_features_for_label = self.csp_transformed_data.get(label, None)
            if csp_features_for_label is not None and i < len(csp_features_for_label):
                combined_features[i, :, :] = csp_features_for_label[i]

        y = np.zeros((X_all.shape[0], len(event_ids)))
        for i, event_id in enumerate(event_ids):
            binary_labels = (event_ids_all == event_id).astype(int)
            y[:, i] = binary_labels

        y_flattened = np.argmax(y, axis=1)
        ftrs = self.featuresarray_load(combined_features)
        
        self.GLOBAL_SHAPE_LENGTH = ftrs.shape[2]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            ftrs, y_flattened, 
            test_size=0.2,  # Changed from train_size=0.9 to test_size=0.2
            random_state=42, 
            stratify=y_flattened
        )

        self.X_train = self.scaler.fit_transform(
            self.X_train.reshape(-1, self.X_train.shape[-1])).reshape(self.X_train.shape)
        self.X_test = self.scaler.transform(
            self.X_test.reshape(-1, self.X_test.shape[-1])).reshape(self.X_test.shape)
        joblib.dump(self.scaler, 'scaler.pkl')
        print("Scaler saved successfully.")
        print(f"Training samples per class: {np.bincount(self.y_train)}")
        print(f"Testing samples per class: {np.bincount(self.y_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def create_dqn_policy(self):
        class KerasDQNPolicy(DQNPolicy):
            def __init__(self_, *args, **kwargs):
                super(KerasDQNPolicy, self_).__init__(*args, **kwargs)
                self_.keras_model = Sequential([
                    Reshape((self.GLOBAL_SHAPE_LENGTH, self.ncomp)),
                    BatchNormalization(),

                    Conv1D(32, kernel_size=3),
                    PReLU(),
                    BatchNormalization(),

                    
                    SpatialDropout1D(0.1),

                    Conv1D(32, kernel_size=3),
                    BatchNormalization(),
                    PReLU(),
                    
                    SpatialDropout1D(0.1),

                    LSTM(32, activation='tanh', recurrent_regularizer=l1_l2(l1=0.01, l2=0.01),return_sequences=True),
                    BatchNormalization(),
                    GlobalMaxPooling1D(),
                    BatchNormalization(),
                    Dense(units=64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
                    BatchNormalization(),
                    
                    Dropout(0.1),
                    Dense(units=self.num_outputs, activation='linear')
                ])
            
            def q_values(self_, obs):
                return self_.keras_model.predict(np.array(obs))
        
        return KerasDQNPolicy

    def create_environment(self):
        class Plasticity(gym.Env):
            def __init__(self_, images_per_episode=1, dataset=(self.X_train, self.y_train), 
                        random=True):
                super(Plasticity, self_).__init__()
                self_.action_space = gym.spaces.Discrete(self.num_outputs)
                self_.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(10, 21),  # 10 timesteps, 21 features
                    dtype=np.float32
                )
                self_.images_per_episode = images_per_episode
                self_.step_count = 0
                self_.x, self_.y = dataset
                self_.random = random
                self_.dataset_idx = 0

            def step(self_, action):
                done = False
                reward = self_._calculate_reward(action)
                obs = self_._next_obs()
                self_.step_count += 1
                if self_.step_count >= self_.images_per_episode:
                    done = True
                return obs, reward, done, {}

            def reset(self_):
                self_.step_count = 0
                return self_._next_obs()

            def _next_obs(self_):
                if self_.random:
                    next_obs_idx = random.randint(0, len(self_.x) - 1)
                    self_.expected_action = int(self_.y[next_obs_idx])
                    obs = self_.x[next_obs_idx]
                else:
                    obs = self_.x[self_.dataset_idx]
                    self_.expected_action = int(self_.y[self_.dataset_idx])
                    self_.dataset_idx = (self_.dataset_idx + 1) % (len(self_.x))
                return obs.reshape(10, 21)  # Reshape to match observation space

            def _calculate_reward(self_, action):
                return 1.0 if action == self_.expected_action else -2.0

        return Plasticity

    def setup_training(self, env):
        model = DQN(
            self.create_dqn_policy(),
            env,
            verbose=0,
            learning_rate=0.0055,
            buffer_size=50000,
            learning_starts=100,
            batch_size=32,
            gamma=0.99,
            train_freq=4,
            target_update_interval=200,
            exploration_fraction=1,
            exploration_final_eps=0.01,
            tensorboard_log="./dqn_plasticity_tensorboard/"
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path='./models/',
            name_prefix='dqn_plasticity'
        )

        class CustomCallback(BaseCallback):
            def __init__(self_, verbose=0):
                super(CustomCallback, self_).__init__(verbose)

            def _on_step(self_) -> bool:
                if self_.n_calls % 1000 == 0:
                    print(f"Step: {self_.n_calls}")
                return True

        return model, [checkpoint_callback, CustomCallback()]



