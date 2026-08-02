import board
import busio
import adafruit_ssd1306

i2c = busio.I2C(board.SCL, board.SDA)
oled = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c)

def draw_sprite(x, y, l, h, sprite_bin);
    for i in range(0,l):
        for j in range(0,h):
            if sprite_bin[i + j*l] > 0.5
                oled.pixel(i,j,1)
    oled.show()
            
def test():
    oled.fill(0)
    oled.show()
    oled.text("Hello, BANANANANANANAN", 0, 0, 1)
    oled.show()
