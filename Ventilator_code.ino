#include<Servo.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 20, 4); // set the LCD address to 0x27 for a 16 chars and 2 line display




Servo m;

#define sensor A0
#define button A1

void setup()
{
  Serial.begin(9600);
  m.attach(3);
  lcd.init();
  delay(100);
  lcd.backlight();
  delay(100);
  //lcd.begin(16, 2);
  delay(100);
  lcd.print("Welecome");
  Serial.println("Welecome");
  delay(1000);
  pinMode(A0, INPUT);
  pinMode(button, INPUT);
  
  
}

int t = 0;

void loop()
{
  int c = analogRead(button);
  Serial.print("Button :");
  Serial.println(c);
  delay(100);
  if (c <20)
  {
    lcd.clear();
    delay(100);
    t = 1;
    lcd.print("Start");
    m.write(0);
    int val = analogRead(A0);
    int val_new = map(val, 0, 1023, 0, 120);
    lcd.setCursor(0, 1);
    lcd.print(" Volume = ");
    int per_val = map(val, 0, 1023, 0, 100);
    lcd.print(per_val);
    lcd.print("%");
    //Serial.println(val_new);
    delay(1000);
    m.write(val_new);
    delay(1000);
  }
  else
  {
//    lcd.clear();
//    delay(100);
//    lcd.print("Stop");
//    delay(500);
    lcd.clear();
    delay(100);
    lcd.print("Press Button");
    lcd.setCursor(0,1);
    lcd.print("To Start");
    delay(1000);
    // Serial.println("Stop");
  }

}
