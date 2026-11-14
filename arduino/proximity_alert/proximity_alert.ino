// Pinpoint demo build: proximity alert for blind and Deaf users.
//
// Wiring (Grove Base Shield) - must match backend/projects/proximity_alert.yaml:
//   Ultrasonic Distance Sensor -> D7
//   Buzzer                     -> D6
//   LCD RGB Backlight          -> any I2C port
//
// Behaviour:
//   > 100 cm     silent, LCD green
//   40-100 cm    slow beeps, LCD yellow
//   < 40 cm      fast beeps, LCD flashing red
// Serial (9600 baud), every 200 ms:  "dist=34 lcd=1"
//   dist = 0 means the sensor did not answer (likely not on D7)
//   lcd  = 1 if the LCD answers on I2C, 0 if not
//
// Libraries (Arduino Library Manager):
//   "Grove Ultrasonic Ranger" (Seeed, provides Ultrasonic.h)
//   "Grove - LCD RGB Backlight" (Seeed, provides rgb_lcd.h)
// The sketch is flashed BEFORE the demo; the build is plugging only.

#include <Wire.h>
#include "Ultrasonic.h"
#include "rgb_lcd.h"

const int ULTRASONIC_PIN = 7;
const int BUZZER_PIN = 6;
const uint8_t LCD_I2C_ADDR = 0x3E;   // text controller on the Grove LCD RGB Backlight

const long FAR_CM = 100;
const long NEAR_CM = 40;
const unsigned long MEASURE_MS = 50;
const unsigned long REPORT_MS = 200;
const unsigned long LCD_MS = 200;
const unsigned long LCD_PROBE_MS = 1000;
const unsigned long BEEP_LEN_MS = 40;
const unsigned long SLOW_BEEP_MS = 600;
const unsigned long FAST_BEEP_MS = 150;
const uint32_t ECHO_TIMEOUT_US = 25000;  // ~4 m; keeps the loop fast when no sensor is attached

Ultrasonic ultrasonic(ULTRASONIC_PIN);
rgb_lcd lcd;

long distanceCm = 0;
bool lcdPresent = false;
unsigned long lastMeasure = 0, lastReport = 0, lastLcd = 0, lastProbe = 0;
unsigned long lastBeepStart = 0;
bool buzzerOn = false;
bool flashOn = false;

bool probeLcd() {
  Wire.beginTransmission(LCD_I2C_ADDR);
  return Wire.endTransmission() == 0;
}

void setup() {
  Serial.begin(9600);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  Wire.begin();
  lcdPresent = probeLcd();
  if (lcdPresent) {
    lcd.begin(16, 2);
  }
}

void updateBuzzer(unsigned long now) {
  unsigned long interval = 0;
  if (distanceCm > 0 && distanceCm < NEAR_CM) interval = FAST_BEEP_MS;
  else if (distanceCm > 0 && distanceCm <= FAR_CM) interval = SLOW_BEEP_MS;

  if (interval == 0) {
    if (buzzerOn) { digitalWrite(BUZZER_PIN, LOW); buzzerOn = false; }
    return;
  }
  if (!buzzerOn && now - lastBeepStart >= interval) {
    digitalWrite(BUZZER_PIN, HIGH);
    buzzerOn = true;
    lastBeepStart = now;
  } else if (buzzerOn && now - lastBeepStart >= BEEP_LEN_MS) {
    digitalWrite(BUZZER_PIN, LOW);
    buzzerOn = false;
  }
}

void updateLcd() {
  if (!lcdPresent) return;
  if (distanceCm <= 0) {
    lcd.setRGB(40, 40, 40);
  } else if (distanceCm > FAR_CM) {
    lcd.setRGB(0, 200, 0);
  } else if (distanceCm >= NEAR_CM) {
    lcd.setRGB(220, 160, 0);
  } else {
    flashOn = !flashOn;
    if (flashOn) lcd.setRGB(255, 0, 0); else lcd.setRGB(0, 0, 0);
  }
  lcd.setCursor(0, 0);
  lcd.print("Distance:       ");
  lcd.setCursor(0, 1);
  if (distanceCm <= 0) {
    lcd.print("no reading      ");
  } else {
    char line[17];
    snprintf(line, sizeof(line), "%-4ld cm  %-6s", distanceCm,
             distanceCm > FAR_CM ? "clear" : (distanceCm >= NEAR_CM ? "near" : "CLOSE"));
    lcd.print(line);
  }
}

void loop() {
  unsigned long now = millis();

  if (now - lastMeasure >= MEASURE_MS) {
    lastMeasure = now;
    distanceCm = ultrasonic.MeasureInCentimeters(ECHO_TIMEOUT_US);
  }

  // LCD may be plugged in after power-up: re-probe and initialise when it appears.
  if (now - lastProbe >= LCD_PROBE_MS) {
    lastProbe = now;
    bool present = probeLcd();
    if (present && !lcdPresent) lcd.begin(16, 2);
    lcdPresent = present;
  }

  updateBuzzer(now);

  if (now - lastLcd >= LCD_MS) {
    lastLcd = now;
    updateLcd();
  }

  if (now - lastReport >= REPORT_MS) {
    lastReport = now;
    Serial.print("dist=");
    Serial.print(distanceCm);
    Serial.print(" lcd=");
    Serial.println(lcdPresent ? 1 : 0);
  }
}
