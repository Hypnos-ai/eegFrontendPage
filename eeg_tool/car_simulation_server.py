from flask import Flask, render_template, jsonify
import threading
import queue
import time

app = Flask(__name__)

# Global command queue for communication between DQN and simulation
command_queue = queue.Queue()

# Car state
car_state = {
    'x': 250,  # Starting x position
    'y': 250,  # Starting y position
    'speed': 0,
    'last_command': 'S'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/command/<cmd>')
def receive_command(cmd):
    command_queue.put(cmd)
    car_state['last_command'] = cmd
    return jsonify({'status': 'ok'})

@app.route('/state')
def get_state():
    return jsonify(car_state)

def update_car_state():
    while True:
        try:
            # Get latest command if available
            try:
                cmd = command_queue.get_nowait()
            except queue.Empty:
                cmd = car_state['last_command']

            # Update car position based on command
            speed = 5  # pixels per update
            
            # Direct movement based on command
            if cmd == 'L':
                car_state['x'] = max(0, car_state['x'] - speed)  # Move left
            elif cmd == 'R':
                car_state['x'] = min(500, car_state['x'] + speed)  # Move right
            elif cmd == 'F':
                car_state['y'] = max(0, car_state['y'] - speed)  # Move up
            elif cmd == 'B':
                car_state['y'] = min(500, car_state['y'] + speed)  # Move down

            time.sleep(0.05)  # 20 FPS update rate
        except Exception as e:
            print(f"Error in car update loop: {e}")
            time.sleep(0.05)

if __name__ == '__main__':
    # Start car update thread
    update_thread = threading.Thread(target=update_car_state, daemon=True)
    update_thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False) 