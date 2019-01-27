#include <Stepper.h>
#include <SPI.h>
#include <SD.h>
#include <Servo.h>

// Stepper Motor Info
const int stepsPerRev = 200;
Stepper myStepperA(stepsPerRev, 6, 7, 8, 9);
Stepper myStepperB(stepsPerRev, 2, 3, 4, 5);
const float StepInc = 1;
const byte StepDelay = 0;
const int LineDelay = 0;
const float StepsPerMillimeterA = 6.3;  // CRITICAL
const float StepsPerMillimeterB = 6.3;  // CRITICAL
const byte speed = 30;

// Servo Info
Servo penServo;
byte penUpAngle = 30;
byte penDownAngle = 90;
int liftDelay = 250;

// Absolute Limits
const float Xmin = 0;
const float Xmax = 1127.13;     // Distance between motors CRITICAL
const float Ymin = 0;
const float Ymax = 400;     // Arbitrary bottom limit

// Working Area
const float Xoffset = 0;   // Offset from left edge
const float Yoffset = 0;  // Offset from top (determined experimentally)

// Current Lengths
float lA;
float lB;

// Starting Coordinates
const float X0 = 330;
const float Y0 = 330;

byte csPin = 10;
float nextX, nextY;
char fileName[] = "code.txt";

File codeFile;

void setup() {
  Serial.begin(9600); // Start Serial Communication

  Serial.print("Loading SD card... ");
  if (!SD.begin(csPin)) {
    Serial.println("Error!");
    while (1) {
      Serial.println("FUCK");
      delay(1000);
    }
  }
  Serial.println("Done.");

  codeFile = SD.open(fileName);

  // Initialize pen to X0, Y0
  lA = getLA(X0, Y0);
  lB = getLB(X0, Y0);

  Serial.print("Starting position: ");
  Serial.print(lA);
  Serial.print(", ");
  Serial.println(lB);

  // Set the stepper motor speeds
  myStepperA.setSpeed(speed);
  myStepperB.setSpeed(speed);

  // Attach the pen-lift servo
  penServo.attach(A5);
  penServo.write(penDownAngle);

  // Leave some time to move away from machine
  delay(2000);

  if (codeFile) {   // file opened successfully
    while (codeFile.available()) {
      String line = "";
      while (codeFile.peek() != '\n') // read until the end of the line
        line.concat(char(codeFile.read()));

      codeFile.read();  // flush the new-line character

      if (line.charAt(0) == 'X') { // Movement Instruction
        nextX = line.substring(line.indexOf('X') + 1, line.indexOf('Y')).toFloat();
        nextY = line.substring(line.indexOf('Y') + 1).toFloat();
        Serial.print("The coordinates are: ");
        Serial.print(nextX);
        Serial.print(", ");
        Serial.println(nextY);
        go(getLA(nextX, nextY), getLB(nextX, nextY));
        //delay(100);
      }
      else if (line.charAt(0) == 'P') { // pen lift command
        if (line.charAt(1) == '0') { // lower pen
          penServo.write(penDownAngle);
          delay(liftDelay);
        }
        else if (line.charAt(1) == '1') { // raise pen
          penServo.write(penUpAngle);
          delay(liftDelay);
        }
      }
      else {
        /*  Future commands can be included by adding
            more "else if" statements. The hash ('#')
            character must be reserved for comments.
        */
      }

    }
    codeFile.close();

  }
  else {
    Serial.println("Error opening instruction file!");
    while (1) {
      Serial.println("FUCK");
      delay(1000);
    }
  }
  penServo.write(penUpAngle);
  go(getLA(X0, Y0), getLB(X0, Y0));

}

void loop()
{

}

// Go to the location given by absolute length A and B
void go(float a, float b) {
  Serial.print("Going to ");
  Serial.print(a);
  Serial.print(", ");
  Serial.println(b);
  //  Convert coordinates to steps
  float a0 = lA;
  float b0 = lB;

  //  Let's find out the change for the coordinates
  long da = abs(a * StepsPerMillimeterA - a0 * StepsPerMillimeterA);
  long db = abs(b * StepsPerMillimeterB - b0 * StepsPerMillimeterB);
  int sa = a0 < a ? StepInc : -StepInc; // set + or - direction
  int sb = b0 < b ? StepInc : -StepInc; // set + or - direction

  unsigned long i;
  long over = 0;

  if (da > db) {
    for (i = 0; i < da; ++i) {
      myStepperA.step(sa);
      over += db;
      if (over >= da) {
        over -= da;
        myStepperB.step(sb);
      }
      delay(StepDelay);
    }
  }
  else {
    for (i = 0; i < db; ++i) {
      myStepperB.step(sb);
      over += da;
      if (over >= db) {
        over -= db;
        myStepperA.step(sa);
      }
      delay(StepDelay);
    }
  }

  //  Update the positions
  lA = a;
  lB = b;

  delay(LineDelay);
}

// Returns the necessary length of A to reach point x,y in the drawing space
float getLA(float x, float y) {
  return sqrt(pow(x + Xoffset, 2) + pow(y + Yoffset, 2));
}

// Returns the necessary length of B to reach point x,y in the drawing space
float getLB(float x, float y) {
  return sqrt(pow(Xmax - x - Xoffset, 2) + pow(y + Yoffset, 2));
}
