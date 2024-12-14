#include <SPI.h>
#include <WiFiNINA.h>
#include "DeviceDriverSet_xxx0.h"

// WiFi credentials
char ssid[] = "Fios-N7K8r";        // your network SSID
char pass[] = "bet77tripe87owl";     // your network password

WiFiServer server(80);

DeviceDriverSet_Motor AppMotor;

enum SmartRobotCarMotionControl
{
  Forward,       //(1)
  Backward,      //(2)
  Left,          //(3)
  Right,         //(4)
  LeftForward,   //(5)
  LeftBackward,  //(6)
  RightForward,  //(7)
  RightBackward, //(8)
  stop_it        //(9)
};

struct Application_xxx
{
  SmartRobotCarMotionControl Motion_Control;
};

Application_xxx Application_SmartRobotCarxxx0;

void ApplicationFunctionSet_SmartRobotCarMotionControl(SmartRobotCarMotionControl direction, uint8_t speed)
{
  switch (direction)
  {
    case Forward:
      AppMotor.DeviceDriverSet_Motor_control(direction_just, speed,
                                           direction_just, speed, control_enable);
      break;
    case Backward:
      AppMotor.DeviceDriverSet_Motor_control(direction_back, speed,
                                           direction_back, speed, control_enable);
      break;
    case Left:
      AppMotor.DeviceDriverSet_Motor_control(direction_just, speed,
                                           direction_back, speed, control_enable);
      break;
    case Right:
      AppMotor.DeviceDriverSet_Motor_control(direction_back, speed,
                                           direction_just, speed, control_enable);
      break;
    case LeftForward:
      AppMotor.DeviceDriverSet_Motor_control(direction_just, speed,
                                           direction_just, speed/2, control_enable);
      break;
    case LeftBackward:
      AppMotor.DeviceDriverSet_Motor_control(direction_back, speed,
                                           direction_back, speed/2, control_enable);
      break;
    case RightForward:
      AppMotor.DeviceDriverSet_Motor_control(direction_just, speed/2,
                                           direction_just, speed, control_enable);
      break;
    case RightBackward:
      AppMotor.DeviceDriverSet_Motor_control(direction_back, speed/2,
                                           direction_back, speed, control_enable);
      break;
    case stop_it:
      AppMotor.DeviceDriverSet_Motor_control(direction_void, 0,
                                           direction_void, 0, control_enable);
      break;
    default:
      break;
  }
}

void setup()
{
  Serial.begin(9600);
  
  // Initialize motor
  AppMotor.DeviceDriverSet_Motor_Init();
  digitalWrite(PIN_Motor_STBY, HIGH); // Enable motors
  
  // Connect to WiFi
  Serial.print("Connecting to WiFi");
  int status = WL_IDLE_STATUS;
  while (status != WL_CONNECTED) {
    status = WiFi.begin(ssid, pass);
    Serial.print(".");
    delay(1000);
  }
  Serial.println("\nConnected to WiFi");
  
  server.begin();
  
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);
}

void loop()
{
  WiFiClient client = server.available();
  if (client) {
    String currentLine = "";
    while (client.connected()) {
      if (client.available()) {
        char c = client.read();
        if (c == '\n') {
          if (currentLine.length() == 0) {
            client.println("HTTP/1.1 200 OK");
            client.println("Content-type:text/html");
            client.println();
            
            // Web interface
            client.println("<!DOCTYPE html><html><head>");
            client.println("<meta name='viewport' content='width=device-width, initial-scale=1'>");
            client.println("<style>");
            client.println("body { font-family: Arial; text-align: center; }");
            client.println(".button { padding: 20px 40px; font-size: 24px; margin: 10px; }");
            client.println("</style></head><body>");
            client.println("<h1>Robot Car Control</h1>");
            
            client.println("<p><a href='/F'><button class='button'>Forward</button></a></p>");
            client.println("<p>");
            client.println("<a href='/L'><button class='button'>Left</button></a>");
            client.println("<a href='/S'><button class='button'>Stop</button></a>");
            client.println("<a href='/R'><button class='button'>Right</button></a>");
            client.println("</p>");
            client.println("<p><a href='/B'><button class='button'>Backward</button></a></p>");
            
            client.println("</body></html>");
            client.println();
            break;
          } else {
            currentLine = "";
          }
        } else if (c != '\r') {
          currentLine += c;
        }

        // Process commands
        if (currentLine.endsWith("GET /F")) {
          ApplicationFunctionSet_SmartRobotCarMotionControl(Forward, 50);
        }
        else if (currentLine.endsWith("GET /B")) {
          ApplicationFunctionSet_SmartRobotCarMotionControl(Backward, 50);
        }
        else if (currentLine.endsWith("GET /L")) {
          ApplicationFunctionSet_SmartRobotCarMotionControl(Left, 50);
        }
        else if (currentLine.endsWith("GET /R")) {
          ApplicationFunctionSet_SmartRobotCarMotionControl(Right, 50);
        }
        else if (currentLine.endsWith("GET /S")) {
          ApplicationFunctionSet_SmartRobotCarMotionControl(stop_it, 0);
        }
      }
    }
    client.stop();
  }
}