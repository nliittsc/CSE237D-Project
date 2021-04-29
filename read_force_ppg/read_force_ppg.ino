#include <Wire.h> //For I2C/SMBus
#include "MAX30105.h"

MAX30105 particleSensor;

void setup(){
  Wire.begin(); // join i2c bus (address optional for master)
  Serial.begin(57600);  // 57600start serial for output

  short data = readDataFromSensor(0x04);
  // Initialize sensor
  particleSensor.begin(Wire, I2C_SPEED_STANDARD);
  
  //Setup to sense a nice looking saw tooth on the plotter
  byte ledBrightness = 0x1F; //Options: 0=Off to 255=50mA     //LED Pulse Amplitude Configuration
  //0x1F = 6.4mA, 0x02 = 0.4mA, 0x7F = 25.4mA, 0xFF= 50.0mA

  byte sampleAverage = 8; //Options: 1, 2, 4, 8, 16, 32
  byte ledMode = 3; //Options: 1 = Red only, 2 = Red + IR, 3 = Red + IR + Green
  int sampleRate = 50; //Options: 50, 100, 200, 400, 800, 1000, 1600, 3200 samples per second
  int pulseWidth = 411; //Options: 69, 118, 215, 411
  int adcRange = 16384; //Options: 2048, 4096, 8192, 16384
  //18 bit resolution
  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange); //Configure sensor with these settings

}

void loop(){
    byte i2cAddress = 0x04; // Slave address (SingleTact), default 0x04
    short data = readDataFromSensor(i2cAddress);
    float fsrVoltage;       // the analog reading converted to voltage
    float parVoltage;       // the analog reading converted to voltage
    float force;
    float vout;
    float voutppg;
    String outt;
  
    fsrVoltage = map(data, 0, 1022, 0, 2000);
    vout = fsrVoltage/1000;

    if (vout < 0.74){ // try with 1
      force = 560*vout-296.8;
    }else{
      force = 230*vout-30;
    }

    if (force < 0){
      vout = 0;
    }
    
    parVoltage = particleSensor.getRed();
    voutppg = parVoltage/100;
    outt = String(vout) + ',' + voutppg;
    Serial.println(outt);
  
    delay(100); // Change this if you are getting values too quickly 
    // Different sampling rate for sending over serial
}


short readDataFromSensor(short address){
  byte i2cPacketLength = 6;//i2c packet length. Just need 6 bytes from each slave
  byte outgoingI2CBuffer[3];//outgoing array buffer
  byte incomingI2CBuffer[6];//incoming array buffer

  outgoingI2CBuffer[0] = 0x01;//I2c read command
  outgoingI2CBuffer[1] = 128;//Slave data offset
  outgoingI2CBuffer[2] = i2cPacketLength;//require 6 bytes

  Wire.beginTransmission(address); // transmit to device 
  Wire.write(outgoingI2CBuffer, 3);// send out command
  byte error = Wire.endTransmission(); // stop transmitting and check slave status
  if (error != 0) return -1; //if slave not exists or has error, return -1
  Wire.requestFrom(address, i2cPacketLength);//require 6 bytes from slave

  byte incomeCount = 0;
  while (incomeCount < i2cPacketLength){    // slave may send less than requested
    if (Wire.available()){
      incomingI2CBuffer[incomeCount] = Wire.read(); // receive a byte as character
      incomeCount++;
    }else{
      delayMicroseconds(10); //Wait 10us 
    }
  }

  short rawData = (incomingI2CBuffer[4] << 8) + incomingI2CBuffer[5]; //get the raw data
  return rawData;
}
