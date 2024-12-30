from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QGroupBox, QCheckBox, QScrollArea, QSlider, QLineEdit, QProgressBar, QFrame, QTabWidget, QMessageBox, QSizePolicy)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
import pyqtgraph as pg
import numpy as np
import mne
import io
import matplotlib.pyplot as plt
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor
from brainflow.board_shim import BoardShim, BoardIds
from scipy.signal import butter, filtfilt
import os
from stable_baselines3.common.callbacks import BaseCallback
from AdaptiveDQN_RLEEGNET import AdaptiveDQNRLEEGNET
import scipy.signal
from session_config import SessionConfig
from pathlib import Path
from eeg_manager import EEGManager
import uuid  # Add this import at the top

class TrainingThread(QThread):
    progress = pyqtSignal(str)  # Signal for progress updates
    finished = pyqtSignal(bool, str)  # Signal for completion (success, message)
    def __init__(self, dqn_model, combined_file, processed_dir, eeg_man, 
                 is_sample=False):
        super().__init__()
        self.dqn_model = dqn_model
        self.combined_file = combined_file
        self.processed_dir = processed_dir
        self.is_sample = is_sample
        self.eeg_man = eeg_man

    def run(self):
        try:
            # Validate task names before training
            if not self.eeg_man.task_names:
                raise ValueError("No tasks defined for training")

            # Setup EEG data
            self.progress.emit("Loading EEG data...")
            raw, events = self.dqn_model.setup_eeg(self.combined_file)
            
            # Process epochs
            self.progress.emit("Processing epochs...")
            X_train, X_test, y_train, y_test = self.dqn_model.process_epochs()
            
            # Debugging output
            self.progress.emit(f"X_train shape: {X_train.shape if X_train is not None else 'None'}")
            self.progress.emit(f"X_test shape: {X_test.shape if X_test is not None else 'None'}")
            self.progress.emit(f"y_train shape: {y_train.shape if y_train is not None else 'None'}")
            self.progress.emit(f"y_test shape: {y_test.shape if y_test is not None else 'None'}")
            
            task_names = list(self.eeg_man.task_names.keys())  
            
            model, callbacks = self.dqn_model.setup_training(task_names)
            
            # Debugging output for model and callbacks
            self.progress.emit(f"Model: {model}")
            self.progress.emit(f"Callbacks: {callbacks}")
            
            # Train model
            self.progress.emit("Starting training...")
            model.learn(total_timesteps=2500, callback=callbacks)

            # Save model
            model_path = os.path.join(
                self.processed_dir,
                'dqn_model_sample.zip' if self.is_sample else 'dqn_model_session.zip'
            )
            model.save(model_path)
            self.finished.emit(True, f"Training complete! Model saved to: {model_path}")
           
            
        except Exception as e:
            self.progress.emit(f"Error during training: {str(e)}")
            self.finished.emit(False, str(e))
        

class MainWindow(QMainWindow):
    def __init__(self, *args):
        super().__init__()
        
        # Initialize dictionaries and configs FIRST
        self.channel_configs = {}  # Store channel configurations with UUIDs
        
        # Define channel colors
        self.channel_colors = {
            'FCz': '#ff69b4',  # Pink
            'C3': '#4CAF50',   # Green
            'Cz': '#2196F3',   # Blue
            'CPz': '#FFC107',  # Amber
            'C2': '#9C27B0',   # Purple
            'C4': '#FF5722',   # Deep Orange
        }
        self.default_channel_colors = [
            '#ff69b4', '#4CAF50', '#2196F3', '#FFC107', 
            '#9C27B0', '#FF5722', '#E91E63', '#00BCD4',
            '#FFEB3B', '#673AB7', '#FF9800', '#03A9F4'
        ]
        
        # Initialize EEG manager next
        self.eeg_man = EEGManager()
        
        # Create visualization panel BEFORE other UI elements
        self.eeg_plot = pg.GraphicsLayoutWidget()
        self.eeg_plot.setBackground('k')
        self.eeg_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.eeg_curves = []
        self.plot_items = {}
        
        # THEN set window properties
        self.setWindowTitle("EEG Analysis Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set black background for entire application
        self.setStyleSheet("""
            QMainWindow {
                background-color: #000000;
            }
            QWidget {
                background-color: #000000;
            }
        """)
        
        # Initialize all UI elements that are used across methods
        self.analyze_btn = None
        self.port_select = QComboBox()
        self.port_select.addItems([f'COM{i}' for i in range(1, 21)])
        self.port_select.setCurrentText('COM9')
        
        self.connect_btn = QPushButton("Connect Board")
        self.connect_btn.clicked.connect(self.toggle_board_connection)
        
        # Create status text area first since it's used in multiple places
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #ff1493;
                border-radius: 5px;
                padding: 5px;
                font-family: monospace;
            }
        """)
        self.status_text.setMaximumHeight(150)
        
        # Create sequence list
        self.sequence_list = QTextEdit()
        self.sequence_list.setReadOnly(True)
        self.sequence_list.setMaximumHeight(100)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Use QHBoxLayout with equal proportions
        layout = QHBoxLayout(main_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Create control panel (left side)
        control_panel = self.create_control_panel()
        control_panel.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        control_panel.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Disable horizontal scroll
        
        # Create right side panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Add visualization panel
        viz_panel = self.create_visualization_panel()
        viz_panel.setMinimumHeight(400)
        viz_panel.setMaximumHeight(500)
        right_layout.addWidget(viz_panel)
        
        # Add status panel
        status_group = QGroupBox("Status Messages")
        status_layout = QVBoxLayout(status_group)
        status_layout.addWidget(self.status_text)
        right_layout.addWidget(status_group)
        
        # Add panels to main layout with equal stretch
        layout.addWidget(control_panel, stretch=1)
        layout.addWidget(right_panel, stretch=1)
        
        # Setup update timer for real-time plotting
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        self.update_timer.start(20)  # 50Hz update rate for smoother display
        
        # Timer setup for task management
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.recording_timeout)
        self.break_timer = QTimer()
        self.break_timer.timeout.connect(self.break_timeout)
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.countdown_timeout)
        
        self.time_remaining = 0
        self.countdown_remaining = 0
        
        # Add cleanup on close
        self.setAttribute(Qt.WA_DeleteOnClose)
        
        # Load last configuration AFTER all UI elements are created
        self.load_last_configuration()
        
        # Add default action if none exist
        if not self.action_inputs:
            self.add_new_action(
                action_name="Default Action",
                description="Perform Default Action",
                repetitions=1
            )
        
        # Add this to your existing initialization
        self.data_buffer = {}  # Buffer for each channel
        self.min_buffer_size = 30  # Minimum points needed (3 * order + 3)
        
    def create_control_panel(self, *args):
        # Create a QScrollArea for vertical scrolling only
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        # Create the panel widget
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins to prevent horizontal scroll
        
        # Update panel stylesheet with black background
        panel.setStyleSheet("""
            QWidget {
                background-color: #000000;
                color: #ffffff;
            }
            QGroupBox {
                border: 2px solid #ff1493;
                border-radius: 5px;
                margin-top: 1ex;
                padding: 10px;
                background-color: #0a0a0a;
            }
            QGroupBox::title {
                color: #ff69b4;
                subcontrol-origin: margin;
                padding: 0 3px;
            }
            QPushButton {
                background: qlineargradient(
                    x1: 0, y1: 0,
                    x2: 1, y2: 0,
                    stop: 0 #4158D0,
                    stop: 0.5 #C850C0,
                    stop: 1 #FFCC70
                );
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(
                    x1: 0, y1: 0,
                    x2: 1, y2: 0,
                    stop: 0 #3147BF,
                    stop: 0.5 #B740AF,
                    stop: 1 #EEBB60
                );
            }
            QPushButton:pressed {
                background: qlineargradient(
                    x1: 0, y1: 0,
                    x2: 1, y2: 0,
                    stop: 0 #2136AE,
                    stop: 0.5 #A6309E,
                    stop: 1 #DDAA50
                );
            }
            QPushButton:disabled {
                background: #333333;
            }
            QLabel {
                color: #ffffff;
            }
            QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
                background-color: #1a1a1a;
                color: white;
                border: 1px solid #ff1493;
                padding: 4px;
                border-radius: 4px;
            }
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #ff1493;
                border-radius: 5px;
                padding: 5px;
            }
            QScrollArea {
                background-color: #000000;
                border: none;
            }
            QFrame {
                background-color: #0a0a0a;
            }
            QProgressBar {
                border: 1px solid #ff1493;
                border-radius: 5px;
                text-align: center;
                height: 25px;
                background-color: #1a1a1a;
            }
            QProgressBar::chunk {
                background-color: #ff1493;
                border-radius: 3px;
            }
            QCheckBox {
                color: #ffffff;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
                background-color: #1a1a1a;
                border: 1px solid #ff1493;
                border-radius: 2px;
            }
            QCheckBox::indicator:checked {
                background-color: #ff1493;
            }
        """)

        # Update status text style
        self.status_text.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #ff1493;
                border-radius: 5px;
                padding: 5px;
                font-family: monospace;
            }
        """)

        # Update sequence list style
        self.sequence_list.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                border: none;
                font-size: 12px;
            }
        """)

        # 1. Board Connection (at the top)
        connection_group = QGroupBox("Board Connection")
        connection_layout = QHBoxLayout(connection_group)
        connection_layout.addWidget(QLabel("Port:"))
        connection_layout.addWidget(self.port_select)
        connection_layout.addWidget(self.connect_btn)
        layout.addWidget(connection_group)

        # 2. Recording Status
        status_group = QGroupBox("Recording Status")
        status_layout = QVBoxLayout(status_group)
        
        self.task_label = QLabel("Current Action: None")
        self.task_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #ffffff;
            padding: 10px;
            background-color: #333333;
            border-radius: 5px;
            border: 1px solid #ff1493;
        """)
        
        # Add instruction label
        self.instruction_label = QLabel("Instruction: ")
        self.instruction_label.setStyleSheet("""
            font-size: 14px;
            color: #ffffff;
            padding: 10px;
            background-color: #333333;
            border-radius: 5px;
            border: 1px solid #ff1493;
        """)
        self.instruction_label.setWordWrap(True)
        
        self.timer_label = QLabel("Time Remaining: --")
        self.timer_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #FF5722;
            padding: 10px;
            background-color: #333333;
            border-radius: 5px;
            border: 1px solid #ff1493;
        """)
        
        self.progress_label = QLabel("Progress: 0/0")
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ff1493;
                border-radius: 5px;
                text-align: center;
                height: 25px;
                background-color: #333333;
            }
            QProgressBar::chunk {
                background-color: #ff1493;
                border-radius: 3px;
            }
        """)
        
        # Add sequence list
        sequence_frame = QFrame()
        sequence_frame.setStyleSheet("""
            QFrame {
                background-color: #333333;
                border: 1px solid #ff1493;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        sequence_layout = QVBoxLayout(sequence_frame)

        sequence_label = QLabel("Upcoming Actions")
        sequence_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #ff69b4;")

        self.sequence_list = QTextEdit()
        self.sequence_list.setReadOnly(True)
        self.sequence_list.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                border: none;
                font-size: 12px;
            }
        """)
        self.sequence_list.setMaximumHeight(100)

        sequence_layout.addWidget(sequence_label)
        sequence_layout.addWidget(self.sequence_list)

        # Add all elements to status layout
        status_layout.addWidget(self.task_label)
        status_layout.addWidget(self.instruction_label)
        status_layout.addWidget(self.timer_label)
        status_layout.addWidget(self.progress_label)
        status_layout.addWidget(self.progress_bar)
        status_layout.addWidget(sequence_frame)  # Add sequence frame
        layout.addWidget(status_group)

        # 3. Session Management
        session_group = QGroupBox("Recording Session")
        session_layout = QVBoxLayout(session_group)
        
        # Session number input
        new_session_widget = QWidget()
        new_session_layout = QHBoxLayout(new_session_widget)
        self.session_num = QSpinBox()
        self.session_num.setMinimum(1)
        new_session_layout.addWidget(QLabel("Session Number:"))
        new_session_layout.addWidget(self.session_num)
        session_layout.addWidget(new_session_widget)
        
        # Session controls only
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        
        # Session controls
        self.start_session_btn = QPushButton("Start Session")
        self.stop_session_btn = QPushButton("Stop Session")
        self.start_session_btn.clicked.connect(self.start_session)
        self.stop_session_btn.clicked.connect(self.stop_session)
        self.stop_session_btn.setEnabled(False)
        
        # Add session buttons to layout
        button_layout.addWidget(self.start_session_btn)
        button_layout.addWidget(self.stop_session_btn)
        session_layout.addWidget(button_widget)
        
        # Previous sessions
        self.session_buttons = QWidget()
        self.session_buttons_layout = QVBoxLayout(self.session_buttons)
        self.update_session_buttons()
        session_layout.addWidget(self.session_buttons)
        
        layout.addWidget(session_group)

        # 4. Action Settings
        action_group = QGroupBox("Action Settings")
        action_layout = QVBoxLayout(action_group)
        
        # Timing settings
        timing_widget = QWidget()
        timing_layout = QVBoxLayout(timing_widget)
        
        # Number of repetitions
        reps_widget = QWidget()
        reps_layout = QHBoxLayout(reps_widget)
        reps_layout.addWidget(QLabel("Number of repetitions for each action:"))
        self.tasks_per_type = QSpinBox()
        self.tasks_per_type.setRange(1, 50)
        self.tasks_per_type.setValue(21)
        reps_layout.addWidget(self.tasks_per_type)
        timing_layout.addWidget(reps_widget)
        
        # Break duration
        break_widget = QWidget()
        break_layout = QHBoxLayout(break_widget)
        break_layout.addWidget(QLabel("Rest time between actions (seconds):"))
        self.break_duration = QSpinBox()
        self.break_duration.setRange(1, 10)
        self.break_duration.setValue(3)
        break_layout.addWidget(self.break_duration)
        timing_layout.addWidget(break_widget)
        
        # Minimum recording time display
        timing_layout.addWidget(QLabel("Minimum recording time: 5 seconds"))
        
        # Add recording time control
        record_time_widget = QWidget()
        record_time_layout = QHBoxLayout(record_time_widget)
        record_time_layout.addWidget(QLabel("Recording time per action (seconds):"))
        self.record_duration = QSpinBox()
        self.record_duration.setRange(5, 60)  # 5 seconds minimum, 60 seconds maximum
        self.record_duration.setValue(5)
        record_time_layout.addWidget(self.record_duration)
        timing_layout.addWidget(record_time_widget)
        
        action_layout.addWidget(timing_widget)
        
        # Action list
        action_list_widget = QWidget()
        self.action_list_layout = QVBoxLayout(action_list_widget)
        self.action_inputs = {}  # Store action inputs
        
        # Add Action button
        add_action_btn = QPushButton("+ Add New Action")
        add_action_btn.clicked.connect(self.add_new_action)
        add_action_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(
                    x1: 0, y1: 0,
                    x2: 1, y2: 0,
                    stop: 0 #4CAF50,
                    stop: 1 #45a049
                );
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
        """)
        
        action_layout.addWidget(add_action_btn)
        action_layout.addWidget(action_list_widget)
        
        # # Add initial actions
        # self.add_new_action()  # Add at least one action by default
        
        layout.addWidget(action_group)

        # Channel controls
        channel_group = self.setup_channel_config()
        layout.addWidget(channel_group)
        
        # Analysis controls (move this BEFORE the Training Controls section)
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        
        # Create analyze button and store as class member
        self.analyze_btn = QPushButton("Analyze Session (CSP)")
        self.analyze_btn.clicked.connect(self.analyze_session)
        self.analyze_btn.setEnabled(False)  # Disabled by default until session is selected
        analysis_layout.addWidget(self.analyze_btn)
        
        layout.addWidget(analysis_group)  # Add analysis group to main layout
        
        # Training Controls
        training_group = QGroupBox("Model Training")
        training_layout = QVBoxLayout(training_group)

        # Training mode selection
        mode_widget = QWidget()
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.addWidget(QLabel("Training Data:"))
        self.training_mode = QComboBox()
        self.training_mode.addItems(["Selected Session Only", "All Recorded Sessions"])
        self.training_mode.setStyleSheet("""
            QComboBox {
                background-color: #333333;
                color: white;
                border: 1px solid #ff1493;
                padding: 4px;
                border-radius: 4px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid none;
                border-right: 5px solid none;
                border-top: 5px solid #ff1493;
                width: 0;
                height: 0;
                margin-right: 5px;
            }
        """)
        mode_layout.addWidget(self.training_mode)
        training_layout.addWidget(mode_widget)

        # Training button
        self.train_dqn_btn = QPushButton("Train DQN-LSTM Model")
        self.train_dqn_btn.clicked.connect(self.train_dqn_model)
        training_layout.addWidget(self.train_dqn_btn)

        # Training progress
        self.training_progress = QTextEdit()
        self.training_progress.setReadOnly(True)
        self.training_progress.setMaximumHeight(100)
        self.training_progress.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #ff1493;
                border-radius: 5px;
                padding: 5px;
                font-family: monospace;
            }
        """)
        training_layout.addWidget(QLabel("Training Progress:"))
        training_layout.addWidget(self.training_progress)

        layout.addWidget(training_group)
        
        # Add Sample Data Visualization group
        sample_group = QGroupBox("Sample Data")
        sample_layout = QVBoxLayout(sample_group)
        
        # Add description
        sample_label = QLabel("View or load pre-recorded sample EEG data:")
        sample_label.setWordWrap(True)
        sample_layout.addWidget(sample_label)
        
        # Sample session selector
        sample_selector = QComboBox()
        sample_files = self.get_sample_files()
        if sample_files:
            sample_selector.addItems([f"Sample {i+1}" for i in range(len(sample_files))])
        sample_layout.addWidget(sample_selector)
        
        # Button container for sample data actions
        sample_buttons = QWidget()
        sample_buttons_layout = QHBoxLayout(sample_buttons)
        
        # View button
        view_sample_btn = QPushButton("View Sample Data")
        view_sample_btn.clicked.connect(lambda: self.view_sample_data(sample_selector.currentIndex()))
        sample_buttons_layout.addWidget(view_sample_btn)
        
        # Load button
        load_sample_btn = QPushButton("Load as Session")
        load_sample_btn.clicked.connect(self.load_sample_as_session)
        sample_buttons_layout.addWidget(load_sample_btn)
        
        sample_layout.addWidget(sample_buttons)
        layout.addWidget(sample_group)
        
        # Add signal controls
        signal_group = QGroupBox("Signal Controls")
        signal_layout = QVBoxLayout(signal_group)

        # Speed control
        speed_widget = QWidget()
        speed_layout = QHBoxLayout(speed_widget)
        speed_layout.addWidget(QLabel("Update Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(self.update_plot_speed)
        speed_layout.addWidget(self.speed_slider)
        signal_layout.addWidget(speed_widget)

        # Filter controls
        filter_widget = QWidget()
        filter_layout = QHBoxLayout(filter_widget)
        filter_layout.addWidget(QLabel("Bandpass Filter:"))
        self.lowcut = QDoubleSpinBox()
        self.lowcut.setRange(0.1, 100.0)
        self.lowcut.setDecimals(1)
        self.lowcut.setSingleStep(0.1)
        self.lowcut.setValue(0.5)  # Default OpenBCI lowcut

        self.highcut = QDoubleSpinBox()
        self.highcut.setRange(1.0, 100.0)
        self.highcut.setDecimals(1)
        self.highcut.setSingleStep(0.1)
        self.highcut.setValue(45.0)  # Default OpenBCI highcut

        filter_layout.addWidget(QLabel("Low:"))
        filter_layout.addWidget(self.lowcut)
        filter_layout.addWidget(QLabel("High:"))
        filter_layout.addWidget(self.highcut)
        signal_layout.addWidget(filter_widget)

        layout.addWidget(signal_group)
        
        # Set fixed width for the scroll area based on parent width
        scroll.setFixedWidth(600)  # Or adjust as needed
        scroll.setWidget(panel)
        
        return scroll
        
    def create_visualization_panel(self, *args):
        """Create the visualization panel with plots"""
        try:
            viz_panel = QWidget()
            layout = QVBoxLayout(viz_panel)
            
            # Create plot widget with size policy
            self.eeg_plot = pg.GraphicsLayoutWidget()
            self.eeg_plot.setBackground('k')
            self.eeg_plot.setSizePolicy(
                QSizePolicy.Expanding, 
                QSizePolicy.Expanding
            )
            layout.addWidget(self.eeg_plot)
            
            # Initialize curves list and plots dict
            self.eeg_curves = []
            self.plot_items = {}  # Store plot items by channel ID
            
            # Initial plot setup
            self.update_plot_channels(self.channel_configs)
            
            return viz_panel
            
        except Exception as e:
            self.add_status_message(f"Error creating visualization panel: {str(e)}")
            return QWidget()

    def update_plot_channels(self, enabled_channels, *args):
        """Update plot display based on enabled channels"""
        try:
            # Clear existing plots
            self.eeg_plot.clear()
            self.eeg_curves = []
            self.plot_items.clear()
            
            # Get enabled channels
            enabled_channels = {
                cid: data for cid, data in enabled_channels.items() 
                if data['enable'].isChecked()
            }
            
            # Create new plots for enabled channels
            for i, (channel_id, data) in enumerate(enabled_channels.items()):
                name = data['input'].text().strip()
                if not name:
                    continue
                    
                # Create plot
                plot = self.eeg_plot.addPlot(row=i, col=0)
                plot.setTitle(name, color='#ff69b4')
                plot.setLabel('left', "μV", color='#ffffff')
                
                # Style the plot
                plot.getAxis('left').setPen('#ffffff')
                plot.getAxis('bottom').setPen('#ffffff')
                plot.showGrid(x=True, y=True, alpha=0.3)
                
                # Show x-axis only on bottom plot
                if i == len(enabled_channels) - 1:
                    plot.setLabel('bottom', "Samples", color='#ffffff')
                else:
                    plot.getAxis('bottom').hide()
                
                # Add curve with channel color
                curve = plot.plot(pen=self.get_channel_color(name))
                self.eeg_curves.append(curve)
                
                # Link x-axis to first plot
                if i > 0:
                    plot.setXLink(self.eeg_plot.getItem(0, 0))
                
                # Store plot reference
                self.plot_items[channel_id] = {
                    'plot': plot,
                    'curve': curve
                }
            
            # Update EEG manager
            channel_names = [
                data['input'].text().strip() 
                for data in enabled_channels.values()
            ]
            self.eeg_man.update_channels(channel_names)
            
            # Force layout update
            self.eeg_plot.updateGeometry()
            
        except Exception as e:
            self.add_status_message(f"Error updating plots: {str(e)}")

    def update_channel_config(self, *args):
        """Update EEG manager with current channel configuration"""
        try:
            # Update plots first
            self.update_plot_channels(self.channel_configs)
            
            # Emit signal that channels have changed
            self.add_status_message("Channel configuration updated")
            
        except Exception as e:
            self.add_status_message(f"Error updating channel config: {str(e)}")

    def resizeEvent(self, event, *args):
        """Handle window resize events"""
        super().resizeEvent(event)
        # Force plot layout update on resize
        if hasattr(self, 'eeg_plot'):
            self.eeg_plot.updateGeometry()

    def add_status_message(self, message, *args):
        self.status_text.append(f"{message}")
        
    def start_session(self, *args):
        """Start a new recording session"""
        try:
            session_num = self.session_num.value()
            
            # Save current configuration
            config = self.save_current_configuration()
            config.save(session_num)
            
            # Update task configuration with current actions
            updated_tasks = {}
            for key, data in self.action_inputs.items():
                action_name = data['input'].text().strip()
                if not action_name:
                    action_name = key
                updated_tasks[action_name] = [
                    f"Perform {action_name}", 
                    self.tasks_per_type.value()
                ]
            
            # Update EEG manager with new task configuration
            self.eeg_man.task_names = updated_tasks
            
            self.eeg_man.initialize_session(session_num)
            
            # Update button states
            self.start_session_btn.setEnabled(False)
            self.stop_session_btn.setEnabled(True)
            
            self.add_status_message(f"Started session {session_num}")
            self.update_session_buttons()
            self.get_next_task()
        except Exception as e:
            self.add_status_message(f"Error starting session: {str(e)}")
            
    def stop_session(self, *args):
        """Stop current session"""
        try:
            # Stop any active timers
            self.recording_timer.stop()
            self.break_timer.stop()
            self.countdown_timer.stop()
            
            # Reset UI elements
            self.timer_label.setText("Time Remaining: --")
            self.progress_bar.setValue(0)
            self.progress_label.setText("Progress: 0/0")
            self.task_label.setText("Current Task: None")
            self.instruction_label.setText("Instruction: ")
            
            # Stop EEG manager
            self.eeg_man.stop_session()
            
            # Reset button states
            self.start_session_btn.setEnabled(True)
            self.stop_session_btn.setEnabled(False)
            
            self.add_status_message("Session stopped")
        except Exception as e:
            self.add_status_message(f"Error stopping session: {str(e)}")
            
    def get_next_task(self, *args):
        task = self.eeg_man.get_next_task()
        if task:
            task_name, instruction = task
            self.task_label.setText(f"Current Task: {task_name}")
            self.instruction_label.setText(f"Instruction: {instruction}")
            
            # Update progress
            current = self.eeg_man.current_task_index
            total = len(self.eeg_man.task_sequence)
            self.progress_label.setText(f"Progress: {current}/{total}")
            self.progress_bar.setValue(int((current / total) * 100))
            
            # Update sequence display
            sequence_text = "Upcoming Tasks:\n"
            remaining_tasks = self.eeg_man.task_sequence[current:]
            for i, (task, instr) in enumerate(remaining_tasks, start=current+1):
                sequence_text += f"{i}. {task}\n"
            self.sequence_list.setText(sequence_text)
            
            # Start countdown
            self.countdown_remaining = 3
            self.countdown_timer.start(1000)
        else:
            self.add_status_message("All tasks completed")
            self.stop_session()
            
    def update_plots(self, *args):
        """Update real-time EEG plot"""
        try:
            if hasattr(self.eeg_man, 'board') and self.eeg_man.board and self.eeg_man.board.is_prepared():
                data = self.eeg_man.get_current_data()
                if data is not None and len(self.eeg_curves) > 0:
                    # Apply filters
                    filtered_data = self.apply_filters(data)
                    
                    # Update each enabled channel's curve
                    data_idx = 0
                    for channel_id, channel_data in self.channel_configs.items():
                        if channel_data['enable'].isChecked():
                            if data_idx < len(filtered_data) and data_idx < len(self.eeg_curves):
                                self.eeg_curves[data_idx].setData(filtered_data[data_idx])
                                data_idx += 1
                            
        except Exception as e:
            print(f"Error updating plots: {e}")
            
    def analyze_session(self, *args):
        """Analyze the current session using CSP"""
        try:
            if self.eeg_man.current_session_num is None:
                self.add_status_message("No session selected for analysis")
                return
                
            self.add_status_message("Starting analysis...")
            
            # Get patterns and class names from EEG Manager
            try:
                patterns, class_names = self.eeg_man.analyze_session(
                    self.eeg_man.current_session_num
                )
                
                # Create analysis window to display CSP patterns
                self.display_csp_patterns(patterns, class_names)
                self.add_status_message("Analysis complete")
                
            except Exception as e:
                self.add_status_message(f"Analysis error: {str(e)}")
                
        except Exception as e:
            self.add_status_message(f"Error analyzing session: {str(e)}")
            
    def get_channel_positions(self, *args):
        """Get channel positions in 2D for topographic plotting"""
        # Standard 10-20 positions for our channels
        positions = {
            'FCz': [0, 0.25],
            'C3': [-0.5, 0],
            'Cz': [0, 0],
            'CPz': [0, -0.25],
            'C2': [0.25, 0],
            'C4': [0.5, 0]
        }
        return positions

    def display_csp_patterns(self, patterns, class_names, *args):
        # Create a new window
        analysis_window = QWidget()
        analysis_window.setWindowTitle("CSP Analysis Results")
        analysis_window.setGeometry(100, 100, 800, 600)  # Reduced window size
        layout = QVBoxLayout(analysis_window)
        
        # Create matplotlib figure with smaller size
        plt.style.use('default')  # Reset style
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # Reduced figure size
        axes = axes.ravel()  # Flatten axes array for easier indexing
        
        for i, (pattern, class_name) in enumerate(zip(patterns, class_names)):
            ax = axes[i]
            mne.viz.plot_topomap(pattern[:, 0], self.eeg_man.raw_data.info, 
                                axes=ax, show=False, contours=0,
                                cmap='RdBu_r', sensors=True,
                                outlines='head', sphere=0.8)  # Added sphere parameter for smaller head size
            ax.set_title(f'CSP Pattern for {class_name}', fontsize=10)  # Reduced font size
        
        plt.tight_layout()
        
        # Convert matplotlib figure to QPixmap
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',  # Reduced DPI
                    facecolor='white', edgecolor='none')
        plt.close()
        
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        buf.close()
        
        # Show in a new window
        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumSize(800, 600)  # Reduced minimum size
        scroll.setWidget(label)
        layout.addWidget(scroll)
        
        # Show window
        analysis_window.show()
        analysis_window.raise_()
        analysis_window.activateWindow()
        
        # Store reference to prevent garbage collection
        self._analysis_window = analysis_window
        
    def countdown_timeout(self, *args):
        """Handle countdown before recording"""
        self.countdown_remaining -= 1
        if self.countdown_remaining > 0:
            self.add_status_message(f"Starting in {self.countdown_remaining}...")
        else:
            self.countdown_timer.stop()
            self.start_recording()
        
    def start_recording(self):
        """Start recording current task"""
        if self.eeg_man.start_recording():
            self.time_remaining = self.record_duration.value()
            self.recording_timer.start(1000)
            self.start_recording_btn.setEnabled(False)
            self.stop_recording_btn.setEnabled(True)
            self.add_status_message("Recording started")
        
    def recording_timeout(self, *args):
        """Handle recording timer"""
        self.time_remaining -= 1
        self.timer_label.setText(f"Time Remaining: {self.time_remaining}s")
        self.timer_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #FF5722;
            padding: 10px;
            background-color: #FBE9E7;
            border-radius: 5px;
        """)
        
        # Update progress bar
        total_tasks = len(self.eeg_man.task_sequence)
        if total_tasks > 0:  # Only calculate progress if there are tasks
            current_task = self.eeg_man.current_task_index
            progress = int((current_task / total_tasks) * 100)
        else:
            self.progress_bar.setValue(0)
            self.progress_label.setText("Progress: 0/0")
        
        if self.time_remaining <= 0:
            self.recording_timer.stop()
            self.eeg_man.stop_recording()
            self.start_break()
        
    def start_break(self, *args):
        """Start break between tasks"""
        self.time_remaining = self.break_duration.value()  # Use configured break duration
        self.break_timer.start(1000)
        self.add_status_message("Break started")
        
    def break_timeout(self, *args):
        """Handle break timer"""
        self.time_remaining -= 1
        self.add_status_message(f"Break: {self.time_remaining}s remaining")
        
        if self.time_remaining <= 0:
            self.break_timer.stop()
            self.get_next_task()
        
    def toggle_board_connection(self, *args):
        """Toggle board connection state"""
        if not self.eeg_man.is_streaming:
            port = self.port_select.currentText()
            if self.eeg_man.initialize_board(port):
                self.connect_btn.setText("Disconnect Board")
                self.add_status_message("Board connected successfully")
            else:
                self.add_status_message("Failed to connect board")
        else:
            if self.eeg_man.stop_stream():
                self.connect_btn.setText("Connect Board")
                self.add_status_message("Board disconnected")
            else:
                self.add_status_message("Failed to disconnect board")
        
    def closeEvent(self, event, *args):
        """Handle application close"""
        if self.eeg_man.is_streaming:
            self.eeg_man.stop_stream()
        event.accept()
        
    def update_channel_visibility(self, *args):
        """Update which channels are visible in the plot"""
        for i, (channel, checkbox) in enumerate(self.channel_checkboxes.items()):
            # Update EEG Manager channel state
            self.eeg_man.set_channel_state(channel, checkbox.isChecked())
            # Update plot visibility
            plot = self.eeg_plot.getItem(i, 0)
            plot.setVisible(checkbox.isChecked())
        
    def update_session_buttons(self, *args):
        """Update the list of session buttons"""
        # Clear existing buttons
        while self.session_buttons_layout.count():
            item = self.session_buttons_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
                
        # Get existing sessions from annotations file
        try:
            with open(self.eeg_man.annotation_file, 'r') as f:
                content = f.read()
                sessions = sorted(list(set([
                    int(x.replace('session', '')) 
                    for x in content.split('\n') 
                    if x.strip().startswith('session')
                ])))
                
            if sessions:
                # Add Delete All button at the top with updated styling
                delete_all_btn = QPushButton("Delete All Sessions")
                delete_all_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #1a1a1a;
                        color: #ffffff;
                        border: 1px solid #ff1493;
                        padding: 8px;
                        border-radius: 4px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #ff1493;
                        color: #ffffff;
                    }
                    QPushButton:pressed {
                        background-color: #cc1177;
                    }
                """)
                delete_all_btn.clicked.connect(self.delete_all_sessions)
                self.session_buttons_layout.addWidget(delete_all_btn)
                
                # Add sessions header with styling
                sessions_label = QLabel("Previous Sessions:")
                sessions_label.setStyleSheet("color: #ff69b4; font-weight: bold; margin-top: 10px;")
                self.session_buttons_layout.addWidget(sessions_label)
                
                for session in sessions:
                    # Create container for session button and delete button
                    session_container = QWidget()
                    container_layout = QHBoxLayout(session_container)
                    container_layout.setContentsMargins(0, 0, 0, 0)
                    container_layout.setSpacing(5)
                    
                    # Session button with updated styling
                    session_btn = QPushButton(f"Session {session}")
                    session_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #1a1a1a;
                            color: #ffffff;
                            border: 1px solid #ff1493;
                            padding: 5px;
                            border-radius: 4px;
                        }
                        QPushButton:hover {
                            background-color: #333333;
                        }
                        QPushButton:pressed {
                            background-color: #444444;
                        }
                    """)
                    session_btn.clicked.connect(lambda checked, s=session: self.select_session(s))
                    container_layout.addWidget(session_btn)
                    
                    # Delete button with updated styling
                    delete_btn = QPushButton("×")
                    delete_btn.setFixedWidth(30)
                    delete_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #1a1a1a;
                            color: #ff1493;
                            border: 1px solid #ff1493;
                            border-radius: 4px;
                            font-weight: bold;
                            font-size: 16px;
                        }
                        QPushButton:hover {
                            background-color: #ff1493;
                            color: #ffffff;
                        }
                        QPushButton:pressed {
                            background-color: #cc1177;
                        }
                    """)
                    delete_btn.clicked.connect(lambda checked, s=session: self.delete_session(s))
                    container_layout.addWidget(delete_btn)
                    
                    self.session_buttons_layout.addWidget(session_container)
                
        except Exception as e:
            print(f"Error loading sessions: {e}")

    def delete_session(self, session_num, *args):
        """Delete a session"""
        try:
            # Create custom styled message box
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setWindowTitle('Delete Session')
            msg_box.setText(f'Are you sure you want to delete Session {session_num}?')
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.No)
            
            # Style the message box
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #1a1a1a;
                }
                QMessageBox QLabel {
                    color: #ffffff;
                    font-size: 12px;
                    padding: 10px;
                }
                QPushButton {
                    background-color: #1a1a1a;
                    color: #ffffff;
                    border: 1px solid #ff1493;
                    padding: 5px 15px;
                    border-radius: 4px;
                    min-width: 60px;
                }
                QPushButton:hover {
                    background-color: #ff1493;
                }
                QPushButton:pressed {
                    background-color: #cc1177;
                }
            """)
            
            if msg_box.exec_() == QMessageBox.Yes:
                self.eeg_man.delete_session(session_num)
                self.update_session_buttons()
                self.add_status_message(f"Deleted Session {session_num}")
                
                if session_num == self.eeg_man.current_session_num:
                    self.analyze_btn.setEnabled(False)
                    self.eeg_man.current_session_num = None
                    
        except Exception as e:
            self.add_status_message(f"Error deleting session: {str(e)}")

    def delete_all_sessions(self, *args):
        """Delete all sessions"""
        try:
            # Create custom styled message box
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle('Delete All Sessions')
            msg_box.setText('Are you sure you want to delete ALL sessions?\nThis will delete all recorded data and cannot be undone.')
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.No)
            
            # Style the message box
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #1a1a1a;
                }
                QMessageBox QLabel {
                    color: #ffffff;
                    font-size: 12px;
                    padding: 10px;
                }
                QPushButton {
                    background-color: #1a1a1a;
                    color: #ffffff;
                    border: 1px solid #ff1493;
                    padding: 5px 15px;
                    border-radius: 4px;
                    min-width: 60px;
                }
                QPushButton:hover {
                    background-color: #ff1493;
                }
                QPushButton:pressed {
                    background-color: #cc1177;
                }
            """)
            
            if msg_box.exec_() == QMessageBox.Yes:
                self.eeg_man.delete_all_sessions()
                self.update_session_buttons()
                self.analyze_btn.setEnabled(False)
                self.add_status_message("Deleted all sessions")
                
        except Exception as e:
            self.add_status_message(f"Error deleting all sessions: {str(e)}")

    def select_session(self, session_num, *args):
        """Select an existing session for analysis"""
        self.eeg_man.current_session_num = session_num
        self.add_status_message(f"Selected Session {session_num}")
        # Enable analyze button only when a session is selected
        self.analyze_btn.setEnabled(True)
        
        # Load session configuration
        self.load_session_configuration(session_num)
    
    def load_session_configuration(self, session_num, *args):
        """Load configuration for specific session"""
        config = SessionConfig.load(session_num)
        if config:
            # Clear existing actions
            while self.action_list_layout.count():
                item = self.action_list_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            # Restore action inputs
            self.action_inputs = {}
            for action_key, action_data in config.actions.items():
                self.add_new_action(
                    action_key=action_key,
                    action_name=action_data['name'],
                    description=f"Perform {action_data['name']}",
                    repetitions=action_data['reps'],
                    silent=True
                )
            
            # Restore settings
            self.tasks_per_type.setValue(config.repetitions)
            self.break_duration.setValue(config.break_duration)
            self.record_duration.setValue(config.record_duration)
            self.lowcut.setValue(config.lowcut)
            self.highcut.setValue(config.highcut)
            
            # Restore channel configurations
            self.channel_configs = {}
            for channel_id, channel_data in config.channel_configs.items():
                self.add_channel(
                    name=channel_data['name'],
                    layout=self.channel_list_layout
                )
                self.channel_configs[channel_id]['enable'].setChecked(channel_data['enabled'])
            
            # Update plots with restored channel configurations
            self.update_plot_channels(self.channel_configs)
        else:
            # Add a default action if no configuration is loaded
            self.add_new_action(
                action_name="Default Action",
                description="Perform Default Action",
                repetitions=1
            )
    
    def save_current_configuration(self):
        """Save current configuration"""
        config = SessionConfig()
        
        # Save actions with their current names
        for action_key, data in self.action_inputs.items():
            config.actions[action_key] = {
                'name': data['input'].text().strip(),
                'reps': self.tasks_per_type.value()
            }
        
        # Save other settings
        config.repetitions = self.tasks_per_type.value()
        config.break_duration = self.break_duration.value()
        config.record_duration = self.record_duration.value()
        config.lowcut = self.lowcut.value()
        config.highcut = self.highcut.value()
        
        # Save channel configurations
        config.channel_configs = {
            channel_id: {
                'name': data['input'].text().strip(),
                'enabled': data['enable'].isChecked()
            }
            for channel_id, data in self.channel_configs.items()
        }
        
        return config
    
    def add_new_action(
        self, 
        checked=False,
        action_name=None, 
        description="Perform Default Action", 
        repetitions=1, 
        *args
    ):
        """Add a new action input row"""
        # Set default action name if none provided
        if action_name is None:
            action_name = f"Action {len(self.action_inputs) + 1}"

        # Check if action already exists
        if action_name in [data['input'].text().strip() for data in self.action_inputs.values()]:
            self.add_status_message(f"Action '{action_name}' already exists.")
            return  # Do not add duplicate actions

        # Create container for action row
        action_row = QWidget()
        row_layout = QHBoxLayout(action_row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        # Generate unique ID for this action
        action_id = str(uuid.uuid4())

        # Create name input
        name_input = QLineEdit()
        name_input.setText(action_name)
        name_input.setPlaceholderText("Enter action name")
        row_layout.addWidget(name_input)

        # Create delete button
        delete_btn = QPushButton("×")
        delete_btn.setFixedWidth(30)
        delete_btn.clicked.connect(lambda: self.delete_action(action_id))
        row_layout.addWidget(delete_btn)

        # Store references with UUID as key
        self.action_inputs[action_id] = {
            'widget': action_row,
            'input': name_input,
            'reps': repetitions,
            'display_name': action_name
        }

        # Add the action row to the layout
        self.action_list_layout.addWidget(action_row)
        self.add_status_message(f"Added action: {action_name}")

    def delete_action(self, action_id, *args):
        """Delete an action by its UUID"""
        if action_id in self.action_inputs:
            # Get widget and display name
            action_data = self.action_inputs[action_id]
            widget = action_data['widget']
            display_name = action_data['display_name']
            
            # Remove widget from layout and delete it
            self.action_list_layout.removeWidget(widget)
            widget.deleteLater()
            
            # Remove from our tracking dict
            del self.action_inputs[action_id]
            
            # Update UI
            self.add_status_message(f"Removed {display_name}")
            self.update_eeg_man_tasks()

    def update_eeg_man_tasks(self, *args):
        """Update EEG manager with current tasks"""
        updated_tasks = {}
        for action_id, data in self.action_inputs.items():
            action_name = data['input'].text().strip()
            if not action_name:  # Use default name if empty
                action_name = f"Action {action_id[:6]}"
            updated_tasks[action_name] = [
                f"Perform {action_name}",
                self.tasks_per_type.value()
            ]
        
        # Update EEG manager
        self.eeg_man.task_names = updated_tasks

    def renumber_actions(self, *args):
        """Renumber actions after deletion"""
        new_inputs = {}
        # Start with Action_1 if it exists
        if "Action_1" in self.action_inputs:
            new_inputs["Action_1"] = self.action_inputs["Action_1"]
        
        # Renumber the rest of the actions starting from 2
        current_num = 2
        for old_key, data in sorted(self.action_inputs.items()):
            if old_key != "Action_1":
                new_key = f"Action_{current_num}"
                new_name = f"Action {current_num}"
                new_inputs[new_key] = data
                data['input'].setText(new_name)
                current_num += 1
            
        self.action_inputs = new_inputs
    
    def load_last_configuration(self, *args):
        """Load and apply the most recent configuration"""
        config = SessionConfig.load_last_config()
        if config:
            # Restore action inputs
            self.action_inputs = {}  # Clear existing
            for action_key, action_data in config.actions.items():
                # Ensure all arguments are provided
                self.add_new_action(
                    action_name=action_data['name'],
                    description=f"Perform {action_data['name']}",
                    repetitions=action_data['reps']
                )
            
            # Restore settings
            self.tasks_per_type.setValue(config.repetitions)
            self.break_duration.setValue(config.break_duration)
            self.record_duration.setValue(config.record_duration)
            self.lowcut.setValue(config.lowcut)
            self.highcut.setValue(config.highcut)
        else:
            # Add a default action if no configuration is loaded
            self.add_new_action(
                action_name="Default Action",
                description="Perform Default Action",
                repetitions=1
            )

    def get_sample_files(self):
        """Get list of available sample data files"""
        try:
            # Look specifically for S03.fif in the formatted subfolder of sample_data
            sample_file = os.path.join(self.eeg_man.sample_dir, 'formatted', 'S03.fif')
            #sample_file = os.path.join("C:\\NeuroSync\\sample_data\\formatted", 'S03.fif')
            
            # Debugging output to status message
            self.add_status_message(f"Checking for sample file at: {sample_file}")
            
            if os.path.exists(sample_file):
                self.add_status_message("Sample file found.")
                return [sample_file]
            else:
                self.add_status_message("Sample file not found.")
                return []
            
        except Exception as e:
            self.add_status_message(f"Error getting sample files: {e}")
            return []

    def view_sample_data(self, index, *args):
        """View selected sample data file"""
        try:
            sample_files = self.get_sample_files()
            if not sample_files:
                self.add_status_message("Sample data file not available")
                return
            
            # Load S03.fif
            #self.add_status_message(f"DEBUG setup_eeg: path = {sample_files[0]}")
            raw = mne.io.read_raw_fif(sample_files[0], preload=True)
            
            # Create a new window for the plot
            plot_window = QWidget()
            plot_window.setWindowTitle(f"Sample Data Visualization - {sample_files[0]}")
            plot_window.setGeometry(200, 200, 1200, 800)
            layout = QVBoxLayout(plot_window)
            
            # Add controls
            controls = QWidget()
            controls_layout = QHBoxLayout(controls)
            
            # View type selector
            view_label = QLabel("View Type:")
            view_select = QComboBox()
            view_select.addItems([
                 
                
                "Butterfly",
                "Channel Spectra",
                "ERP Image",
                
                "Topographic Map"
            ])
            controls_layout.addWidget(view_label)
            controls_layout.addWidget(view_select)
            
            # Scale spinner
            scale_label = QLabel("Scale (µV):")
            scale_spin = QSpinBox()
            scale_spin.setRange(1, 1000)
            scale_spin.setValue(50)
            controls_layout.addWidget(scale_label)
            controls_layout.addWidget(scale_spin)
            
            # Duration spinner
            duration_label = QLabel("Window (s):")
            duration_spin = QSpinBox()
            duration_spin.setRange(1, 600)  # Allow up to 10 minutes
            duration_spin.setValue(10)
            duration_spin.setKeyboardTracking(True)  # Enable keyboard input
            duration_spin.setButtonSymbols(QSpinBox.NoButtons)  # Remove arrow buttons
            duration_spin.setStyleSheet("""
                QSpinBox {
                    padding: 5px;
                    width: 60px;
                }
            """)
            controls_layout.addWidget(duration_label)
            controls_layout.addWidget(duration_spin)
            
            # Update button
            update_btn = QPushButton("Update Plot")
            controls_layout.addWidget(update_btn)
            
            layout.addWidget(controls)
            
            def update_plot():
                plt.close('all')
                
                view_type = view_select.currentText()
                if view_type == "Time Series":
                    raw.plot(
                        duration=duration_spin.value(),
                        scalings=dict(eeg=scale_spin.value() * 1e-6),
                        title='EEG Time Series',
                        show=True,
                        block=False,
                        theme='dark'  # MNE's built-in dark theme
                    )
                
                elif view_type == "Power Spectrum":
                    fig = raw.plot_psd(
                        fmax=50,
                        average=True,
                        show=False,
                        dB=True,
                        estimate='power',
                        picks='eeg'  # Explicitly specify EEG channels
                    )
                    # Style the plot
                    ax = fig.axes[0]
                    ax.set_facecolor('black')
                    fig.patch.set_facecolor('black')
                    ax.tick_params(colors='white')
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    ax.title.set_color('white')
                    ax.grid(True, alpha=0.2)  # Add grid for better readability
                    plt.show()
                
                elif view_type == "Butterfly":
                    data = raw.get_data()
                    times = raw.times
                    plt.figure(figsize=(12, 6), facecolor='black')
                    for i in range(data.shape[0]):
                        plt.plot(times, data[i] * 1e6, linewidth=0.5, 
                                alpha=0.7, color=plt.cm.rainbow(i/data.shape[0]))
                    plt.xlabel('Time (s)', color='white')
                    plt.ylabel('Amplitude (μV)', color='white')
                    plt.title('EEG Butterfly Plot', color='white')
                    plt.grid(True, alpha=0.2)
                    ax = plt.gca()
                    ax.set_facecolor('black')
                    ax.tick_params(colors='white')
                    plt.show()
                
                elif view_type == "Channel Spectra":
                    # Plot individual channel spectra
                    fig = raw.plot_psd_topo(
                        tmax=duration_spin.value(),
                        fmax=50,
                        show=False
                    )
                    # Style the plot
                    for ax in fig.axes:
                        ax.set_facecolor('black')
                        ax.tick_params(colors='white')
                        ax.xaxis.label.set_color('white')
                        ax.yaxis.label.set_color('white')
                    fig.patch.set_facecolor('black')
                    plt.show()
                
                elif view_type == "ERP Image":
                    # Create epochs from continuous data
                    events = mne.make_fixed_length_events(
                        raw, duration=duration_spin.value()
                    )
                    epochs = mne.Epochs(
                        raw, events, tmin=0, tmax=duration_spin.value(),
                        baseline=None, preload=True
                    )
                    
                    # Plot ERP image with correct figure handling
                    fig = epochs.plot_image(
                        picks='eeg', combine='mean', show=False,
                        title='ERP Image'
                    )[0]  # Get the first figure from returned tuple
                    
                    # Style the plot
                    fig.set_facecolor('black')
                    for ax in fig.get_axes():  # Use get_axes() instead of axes attribute
                        ax.set_facecolor('black')
                        ax.tick_params(colors='white')
                        ax.xaxis.label.set_color('white')
                        ax.yaxis.label.set_color('white')
                        ax.title.set_color('white')
                    plt.show()
                
                elif view_type == "Evoked Array":
                    # Create epochs and compute evoked response
                    events = mne.make_fixed_length_events(
                        raw, duration=duration_spin.value()
                    )
                    epochs = mne.Epochs(
                        raw, events, tmin=0, tmax=duration_spin.value(),
                        baseline=None, preload=True
                    )
                    evoked = epochs.average()
                    
                    # Plot evoked data in various ways
                    fig = plt.figure(figsize=(15, 10), facecolor='black')
                    gs = fig.add_gridspec(2, 2)
                    
                    # 1. Joint plot (top left)
                    plt.subplot(gs[0, 0])
                    evoked.plot_joint(
                        times=[0.1, 0.2, 0.3],  # Show topomaps at these times
                        show=False
                    )
                    
                    # 2. Topomap sequence (top right)
                    plt.subplot(gs[0, 1])
                    evoked.plot_topomap(
                        times=np.linspace(0, duration_spin.value(), 6),
                        show=False
                    )
                    
                    # 3. Butterfly plot (bottom left)
                    plt.subplot(gs[1, 0])
                    evoked.plot(spatial_colors=True, show=False)
                    
                    # 4. GFP plot (bottom right)
                    plt.subplot(gs[1, 1])
                    evoked.plot_image()
                    
                    # Style all subplots
                    for ax in fig.get_axes():
                        ax.set_facecolor('black')
                        ax.tick_params(colors='white')
                        ax.xaxis.label.set_color('white')
                        ax.yaxis.label.set_color('white')
                        if ax.get_title():
                            ax.title.set_color('white')
                    
                    plt.tight_layout()
                    plt.show()
                
                elif view_type == "Topographic Map":
                    # Create topographic map of signal power
                    data = raw.get_data(start=0, 
                                      stop=int(duration_spin.value() * raw.info['sfreq']))
                    power = np.std(data, axis=1)  # Use standard deviation as power measure
                    mne.viz.plot_topomap(
                        power, raw.info,
                        cmap='RdBu_r',
                        sensors=True,  # Show sensor locations instead of show_names
                        names=raw.ch_names,  # Show channel names
                        show=True
                    )
            
            update_btn.clicked.connect(update_plot)
            
            # Initial plot
            update_plot()
            
            # Show window
            plot_window.show()
            plot_window.raise_()
            plot_window.activateWindow()
            
            # Store reference
            self._plot_window = plot_window
            
        except Exception as e:
            self.add_status_message(f"Error viewing sample data: {str(e)}")

    def apply_channel_config(self, *args):
        """Apply channel configuration"""
        try:
            # Validate all names are unique and not empty
            new_names = [input.text().strip() for input in self.channel_names.values()]
            if len(set(new_names)) != len(new_names) or '' in new_names:
                self.add_status_message("Error: Channel names must be unique and not empty")
                return
            
            # Update plot titles and channel mapping
            for i, (original_name, input_widget) in enumerate(self.channel_names.items()):
                new_name = input_widget.text().strip()
                plot = self.eeg_plot.getItem(i, 0)
                plot.setTitle(new_name)
            
            self.add_status_message("Channel configuration updated")
        except Exception as e:
            self.add_status_message(f"Error applying channel configuration: {str(e)}")

    def update_plot_speed(self, *args):
        """Update the plot refresh rate"""
        speed = self.speed_slider.value()
        interval = int(1000 / speed)
        self.update_timer.setInterval(interval)

    def train_dqn_model(self, *args):
        """Train DQN model on current session data"""
        try:
            if self.eeg_man.current_session_num is None:
                raise ValueError("No session selected. Please select a session first.")
                
            self.training_progress.clear()
            self.training_progress.append("Initializing DQN training...")
            
            # Initialize the model
            dqn_model = AdaptiveDQNRLEEGNET()
            
            # Handle sample data differently
            if self.eeg_man.current_session_num == 0:  # Sample data
                combined_file = os.path.join(self.eeg_man.sample_dir, 'formatted', 'S03.fif')
            else:
                # Regular session - combine files first
                self.training_progress.append("Combining FIF files...")
                try:
                    combined_file = self.eeg_man.combine_fif_files()
                    self.training_progress.append("Files combined successfully")
                except Exception as e:
                    raise Exception(f"Error combining files: {str(e)}")
            
            # Create and start training thread
            self.training_thread = TrainingThread(
                dqn_model, 
                combined_file, 
                self.eeg_man.processed_dir,
                is_sample=(self.eeg_man.current_session_num == 0),
                eeg_man=self.eeg_man
            )
            
            # Connect signals
            self.training_thread.progress.connect(
                lambda msg: self.training_progress.append(msg)
            )
            self.training_thread.finished.connect(
                lambda success, msg: self.handle_training_complete(success, msg)
            )
            
            # Disable train button while training
            self.train_dqn_btn.setEnabled(False)
            
            # Start training in background
            self.training_thread.start()
            
        except Exception as e:
            self.training_progress.append(f"Training error: {str(e)}")

    def handle_training_complete(self, success, message, *args):
        """Handle completion of training thread"""
        self.training_progress.append(message)
        self.train_dqn_btn.setEnabled(True)
        
        if success:
            self.add_status_message("Training completed successfully")
        else:
            self.add_status_message("Training failed - check progress log for details")

    def apply_filters(self, data, *args):
        """Apply bandpass filter to EEG data with buffering"""
        try:
            # Get filter parameters from UI
            lowcut = self.lowcut.value()
            highcut = self.highcut.value()
            
            # Get sampling rate using board_id
            board_id = self.eeg_man.board.get_board_id()
            fs = BoardShim.get_sampling_rate(board_id)
            
            # Design Butterworth bandpass filter
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            order = 4
            b, a = butter(order, [low, high], btype='band')
            
            filtered_data = np.zeros_like(data)
            
            # Process each channel
            for i in range(data.shape[0]):
                # Initialize buffer for this channel if it doesn't exist
                if i not in self.data_buffer:
                    self.data_buffer[i] = []
                    
                # Convert current data to list and extend buffer
                current_data = data[i].tolist()
                self.data_buffer[i].extend(current_data)
                
                # Keep only the last N points to prevent memory issues
                max_buffer = self.min_buffer_size * 2  # Keep twice what we need
                if len(self.data_buffer[i]) > max_buffer:
                    self.data_buffer[i] = self.data_buffer[i][-max_buffer:]
                
                # Apply filter if we have enough data
                if len(self.data_buffer[i]) >= self.min_buffer_size:
                    # Filter the buffered data
                    filtered_buffer = filtfilt(b, a, np.array(self.data_buffer[i]))
                    
                    # Ensure the filtered data length matches the input data length
                    if len(filtered_buffer) >= len(current_data):
                        start_idx = len(filtered_buffer) - len(current_data)
                        filtered_data[i] = filtered_buffer[start_idx:]
                    else:
                        # If filtered data is shorter, use the unfiltered data
                        filtered_data[i] = data[i]
                else:
                    # Not enough data yet, use unfiltered data
                    filtered_data[i] = data[i]
                    
            return filtered_data
            
        except Exception as e:
            self.add_status_message(f"Error applying filters: {str(e)}")
            return data  # Return original data if filtering fails

    def stop_recording(self, *args):
        """Stop current recording"""
        self.recording_timer.stop()
        self.eeg_man.stop_recording()
        self.start_recording_btn.setEnabled(True)
        self.stop_recording_btn.setEnabled(False)
        self.start_break()  # Start break after recording stops
        self.add_status_message("Recording stopped")

    def load_sample_as_session(self, *args):
        """Load and train on sample data"""
        try:
            # Get path to S03.fif in sample_data
            sample_file = os.path.join(self.eeg_man.sample_dir, 'formatted', 'S03.fif')
            if not os.path.exists(sample_file):
                self.add_status_message("Sample data (S03.fif) not found")
                return

            # Load the sample data directly
            #self.add_status_message(f"DEBUG setup_eeg: path = {sample_file}")
            raw = mne.io.read_raw_fif(sample_file, preload=True)
            self.eeg_man.raw_data = raw
            
            # Set default actions for sample data
            self.eeg_man.task_names = {
                "Right_Hand": ["Right Hand Movement", 21],
                "Left_Hand": ["Left Hand Movement", 21],
                "Blinking": ["Blinking", 21],
                "Jaw_Clenching": ["Jaw Clenching", 21],
                "Relax": ["Relax", 21]
            }

            # Populate channels and actions
            self.populate_channels_and_actions()

            # Set as current session and enable analysis
            self.eeg_man.current_session_num = 0  # Use 0 to indicate sample data
            self.analyze_btn.setEnabled(True)
            self.add_status_message("Loaded sample data (S03.fif)")
            
            # Update UI to show sample data is selected
            self.update_session_buttons()
            
        except Exception as e:
            self.add_status_message(f"Error loading sample data: {str(e)}")

    def populate_channels_and_actions(self):
        """Populate channels and actions for the current session"""
        # Clear existing channels
        while self.channel_list_layout.count():
            item = self.channel_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.channel_configs.clear()

        # Add channels
        for channel in self.eeg_man.channel_names:
            self.add_channel(channel, self.channel_list_layout)

        # Clear existing actions
        while self.action_list_layout.count():
            item = self.action_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.action_inputs.clear()

        # Add actions
        for action_name, (description, repetitions) in self.eeg_man.task_names.items():
            self.add_new_action(action_name, description, repetitions)

    def setup_channel_config(self, *args):
        """Setup channel configuration UI with UUID-based channels"""
        try:
            channel_group = QGroupBox("EEG Channels")
            channel_layout = QVBoxLayout(channel_group)
            
            # Add header with better styling
            header = QWidget()
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(5, 5, 5, 5)
            
            # Style the headers
            for label_text in ["Channel Name", "Enable", "Actions"]:
                label = QLabel(label_text)
                label.setStyleSheet("""
                    QLabel {
                        color: #ff69b4;
                        font-weight: bold;
                        font-size: 14px;
                    }
                """)
                header_layout.addWidget(label)
            channel_layout.addWidget(header)
            
            # Channel list container with scroll
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet("""
                QScrollArea {
                    border: 1px solid #666;
                    border-radius: 4px;
                    background: #2d2d2d;
                }
            """)
            
            scroll_widget = QWidget()
            self.channel_list_layout = QVBoxLayout(scroll_widget)
            self.channel_list_layout.setSpacing(5)
            self.channel_list_layout.setContentsMargins(5, 5, 5, 5)
            scroll.setWidget(scroll_widget)
            channel_layout.addWidget(scroll)
            
            # Add default channels
            default_channels = ['FCz', 'C3', 'Cz', 'CPz', 'C2', 'C4']
            for channel in default_channels:
                self.add_channel(channel, self.channel_list_layout)
            
            # Force initial plot update after adding default channels
            self.update_plot_channels(self.channel_configs)
            
            # Add channel button with more visible styling
            add_btn = QPushButton("+ Add New Channel")
            add_btn.clicked.connect(lambda: self.add_channel_and_update(f"Channel_{len(self.channel_configs) + 1}"))
            add_btn.setStyleSheet("""
                QPushButton {
                    background: #4CAF50;
                    color: white;
                    border-radius: 4px;
                    padding: 8px;
                    margin: 8px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background: #45a049;
                }
            """)
            channel_layout.addWidget(add_btn)
            
            return channel_group
            
        except Exception as e:
            self.add_status_message(f"Error setting up channel config: {str(e)}")

    def add_channel_and_update(self, name, *args):
        """Add a channel and update plots"""
        self.add_channel(name, self.channel_list_layout)
        self.update_plot_channels(self.channel_configs)

    def add_channel(self, name, layout, *args):
        """Add a new channel"""
        # Check if channel already exists
        if name in [data['input'].text().strip() for data in self.channel_configs.values()]:
            return  # Do not add duplicate channels

        # Create channel row
        channel_row = QWidget()
        row_layout = QHBoxLayout(channel_row)
        row_layout.setContentsMargins(5, 5, 5, 5)

        # Generate UUID for channel
        channel_id = str(uuid.uuid4())

        # Create name input
        name_input = QLineEdit(name)
        name_input.setPlaceholderText("Channel name")
        row_layout.addWidget(name_input)

        # Create enable/disable checkbox
        enable_box = QCheckBox("Enable")
        enable_box.setChecked(True)
        row_layout.addWidget(enable_box)

        # Create delete button
        delete_btn = QPushButton("×")
        delete_btn.setFixedWidth(30)
        delete_btn.clicked.connect(lambda: self.delete_channel(channel_id))
        row_layout.addWidget(delete_btn)

        # Store channel config
        self.channel_configs[channel_id] = {
            'widget': channel_row,
            'input': name_input,
            'enable': enable_box,
            'name': name
        }

        # Add to layout
        layout.addWidget(channel_row)

    def delete_channel(self, channel_id, *args):
        """Delete a channel by UUID"""
        if channel_id in self.channel_configs:
            channel_data = self.channel_configs[channel_id]
            widget = channel_data['widget']
            name = channel_data['name']
            
            # Remove widget and delete
            widget.setParent(None)
            widget.deleteLater()
            
            # Remove from configs
            del self.channel_configs[channel_id]
            
            self.add_status_message(f"Removed channel: {name}")
            self.update_channel_config()

    def update_channel_config(self, *args):
        """Update EEG manager with current channel configuration"""
        try:
            # Update plots first
            self.update_plot_channels(self.channel_configs)
            
            # Emit signal that channels have changed
            self.add_status_message("Channel configuration updated")
            
        except Exception as e:
            self.add_status_message(f"Error updating channel config: {str(e)}")

    def update_plot_channels(self, enabled_channels, *args):
        """Update plot display based on enabled channels"""
        try:
            # Clear existing plots
            self.eeg_plot.clear()
            self.eeg_curves = []
            self.plot_items.clear()
            
            # Get enabled channels
            enabled_channels = {
                cid: data for cid, data in enabled_channels.items() 
                if data['enable'].isChecked()
            }
            
            # Create new plots for enabled channels
            for i, (channel_id, data) in enumerate(enabled_channels.items()):
                name = data['input'].text().strip()
                if not name:
                    continue
                    
                # Create plot
                plot = self.eeg_plot.addPlot(row=i, col=0)
                plot.setTitle(name, color='#ff69b4')
                plot.setLabel('left', "μV", color='#ffffff')
                
                # Style the plot
                plot.getAxis('left').setPen('#ffffff')
                plot.getAxis('bottom').setPen('#ffffff')
                plot.showGrid(x=True, y=True, alpha=0.3)
                
                # Show x-axis only on bottom plot
                if i == len(enabled_channels) - 1:
                    plot.setLabel('bottom', "Samples", color='#ffffff')
                else:
                    plot.getAxis('bottom').hide()
                
                # Add curve with channel color
                curve = plot.plot(pen=self.get_channel_color(name))
                self.eeg_curves.append(curve)
                
                # Link x-axis to first plot
                if i > 0:
                    plot.setXLink(self.eeg_plot.getItem(0, 0))
                
                # Store plot reference
                self.plot_items[channel_id] = {
                    'plot': plot,
                    'curve': curve
                }
            
            # Update EEG manager
            channel_names = [
                data['input'].text().strip() 
                for data in enabled_channels.values()
            ]
            self.eeg_man.update_channels(channel_names)
            
            # Force layout update
            self.eeg_plot.updateGeometry()
            
        except Exception as e:
            self.add_status_message(f"Error updating plots: {str(e)}")

    def get_channel_color(self, channel_name, *args):
        """Get color for a channel, with fallback to default colors"""
        # First check if channel has a specific color
        if channel_name in self.channel_colors:
            return self.channel_colors[channel_name]
        
        # If not, assign a color from the default list based on channel name hash
        color_index = hash(channel_name) % len(self.default_channel_colors)
        return self.default_channel_colors[color_index]

    def setup_action_buttons(self):
        """Setup buttons for adding and managing actions"""
        add_action_btn = QPushButton("Add Action")
        add_action_btn.clicked.connect(lambda: self.add_new_action(
            action_name=f"Action {len(self.action_inputs) + 1}",
            description="New Action",
            repetitions=1
        ))
        self.action_button_layout.addWidget(add_action_btn)