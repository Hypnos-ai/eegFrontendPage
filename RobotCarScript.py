import socket
import time
import keyboard  # for keyboard control

class RobotCar:
    def __init__(self, ip_address, port=80):
        self.ip = ip_address
        self.port = port
        
    def send_command(self, command):
        try:
            # Create a socket connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.ip, self.port))
            
            # HTTP GET request
            request = f"GET /{command} HTTP/1.1\r\nHost: {self.ip}\r\n\r\n"
            sock.send(request.encode())
            
            # Close the connection
            sock.close()
            print(f"Sent command: {command}")
            
        except Exception as e:
            print(f"Error sending command: {e}")
            
    def forward(self):
        self.send_command('F')
        
    def backward(self):
        self.send_command('B')
        
    def left(self):
        self.send_command('L')
        
    def right(self):
        self.send_command('R')
        
    def stop(self):
        self.send_command('S')

def test_connection(ip_address, port=80):
    print(f"Testing connection to car at {ip_address}:{port}")
    
    commands = ['F', 'B', 'L', 'R', 'S']
    
    for cmd in commands:
        try:
            # Create socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((ip_address, port))
            
            # Send command
            request = f"GET /{cmd} HTTP/1.1\r\nHost: {ip_address}\r\n\r\n"
            sock.send(request.encode())
            
            print(f"Successfully sent command: {cmd}")
            sock.close()
            time.sleep(1)  # Wait between commands
            
        except Exception as e:
            print(f"Error with command {cmd}: {e}")
            return False
    
    return True

def main():
    # Replace with your car's IP address from Serial Monitor
    car_ip = "192.168.1.168"  # CHANGE THIS TO YOUR CAR'S IP
    car = RobotCar(car_ip)
    
    print("Robot Car Control")
    print("----------------")
    print("Use arrow keys to control the car:")
    print("↑ - Forward")
    print("↓ - Backward")
    print("← - Left")
    print("→ - Right")
    print("Space - Stop")
    print("Q - Quit")
    print("----------------")
    
    try:
        while True:
            if keyboard.is_pressed('up'):
                car.forward()
                time.sleep(0.1)  # Delay to prevent too many commands
            elif keyboard.is_pressed('down'):
                car.backward()
                time.sleep(0.1)
            elif keyboard.is_pressed('left'):
                car.left()
                time.sleep(0.1)
            elif keyboard.is_pressed('right'):
                car.right()
                time.sleep(0.1)
            elif keyboard.is_pressed('space'):
                car.stop()
                time.sleep(0.1)
            elif keyboard.is_pressed('q'):
                print("Quitting...")
                break
                
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()



