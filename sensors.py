#!/usr/bin/env python3

import time, colorsys, sys, os
import pickle
import numpy as np
import itertools
import ST7735
try:
    # Transitional fix for breaking change in LTR559
    from ltr559 import LTR559
    ltr559 = LTR559()
except ImportError:
    import ltr559

from bme280 import BME280
from pms5003 import PMS5003, ReadTimeoutError as pmsReadTimeoutError, SerialTimeoutError
from enviroplus import gas
from subprocess import PIPE, Popen
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from fonts.ttf import RobotoMedium as UserFont

INTEGRATORS_FILENAME = "integrators.pyobj"


ltr559.set_light_options(gain=96)
ltr559.set_light_integration_time_ms(400)

# BME280 temperature/pressure/humidity sensor
bme280 = BME280()

# PMS5003 particulate sensor
pms5003 = PMS5003()
time.sleep(1.0)

# Create ST7735 LCD display class
st7735 = ST7735.ST7735(
    port=0,
    cs=1,
    dc=9,
    backlight=12,
    rotation=270,
    spi_speed_hz=10000000
)

# Initialize display
st7735.begin()

WIDTH = st7735.width
HEIGHT = st7735.height

# Set up canvas and font
img = Image.new('RGB', (WIDTH, HEIGHT), color=(0, 0, 0))
draw = ImageDraw.Draw(img)
font = ImageFont.truetype(UserFont, 18)
x_offset = 2
y_offset = 2


# Displays all the text on the 0.96" LCD
def display_everything(values, bars = None, bg=(0, 0, 0)):
    limits = {'temp': [18, 20, 23, 26],
              'hum': [20, 35, 50, 65],
              'pres' : [980, 1005, 1020, 1030],
              'pms010': [2, 5, 10, 20],
              'pms025': [2, 5, 10, 20],
              'pms100': [2, 5, 10, 20] }
    palette = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
    draw.rectangle((0, 0, WIDTH, HEIGHT), bg)
    if bars is not None:
        tri = HEIGHT//3
        draw.rectangle((WIDTH-20, 0*tri, WIDTH, 1*tri), bars[0])
        draw.rectangle((WIDTH-20, 1*tri, WIDTH, 2*tri), bars[1])
        draw.rectangle((WIDTH-20, 2*tri, WIDTH, 3*tri), bars[2])
                
    column_count = 2
    row_count = (len(values) / column_count)
    for i, (variable, value) in enumerate(values.items()):
        x = x_offset + ((WIDTH // column_count) * (i // row_count))
        y = y_offset + ((HEIGHT / row_count) * (i % row_count))
        message = "{} {:.1f}".format(variable[0], value)
        lim = limits[variable]
        rgb = palette[0]
        for j in range(len(lim)):
            if value > lim[j]:
                rgb = palette[j + 1]
        draw.text((x, y), message, font=font, fill=rgb)
    st7735.display(img)

class Anom:
    def __init__(self, dim, decay, creg=None):
        self.k = 1 - decay
        self.N = self.mean = self.SS = 0.0
        self.dim = dim
        self.creg = creg if creg is not None else np.ones(dim, 'float') / 10000.0
    def update(self, v):
        k, N, mean, SS = self.k, self.N, self.mean, self.SS
        N += 1; dv = v - mean
        mean += dv / N
        # careful here! This is symmetric, see
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # In general, don't touch this unless you know what you are doing...
        SS += np.outer(dv, v - mean) 
        self.N, self.mean, self.SS = k * N, mean, k * SS
    def z2(self, v):
        mean = self.mean
        C = self.SS / self.N + np.diag(self.creg)
        iC = np.linalg.inv(C)
        # this is chi-sq / k, with mean 1, var = 2k/k^2 = 2/k, or sd = sqrt(2/k)
        # That is, N(1, sqrt(2/k)*^2) for large k
        z2 = np.einsum("i,ij,j", v - mean, iC, v - mean) / self.dim 
        return z2
    def uz2(self, v):
        self.update(v)
        return self.z2(v)
    
xs = []
v_lngs = []

if os.path.isfile(INTEGRATORS_FILENAME):
        a_act, a_env, a_gpm, a_lng = pickle.load(open(INTEGRATORS_FILENAME, 'rb'))
else:
        a_act = Anom(6, .001) # "nearby activity" half time about 10 minutes (sample per sec)
        a_env = Anom(5, .0002) # temp, hum and gas; an hour
        a_gpm = Anom(6, .00005) # slow-moving pres, env and pms; five hours
        a_lng = Anom(10, .0001) # half time about five days (sample per minute)

for i in itertools.count():
    proximity = ltr559.get_proximity()
    lux = ltr559.get_lux()
    temp = bme280.get_temperature()
    humidity = bme280.get_humidity()
    pressure = bme280.get_pressure()
    pms_obj = pms5003.read()
    gas_obj = gas.read_all()

    # x are for logging and display
    x_gas = tuple([7 - np.log10(x) for x in (gas_obj.oxidising, gas_obj.reducing, gas_obj.nh3)])
    x_pms = tuple([pms_obj.pm_ug_per_m3(x) for x in (1.0, 2.5, 10.0)])

    # v are for anomalies etc
    v_gas = np.array(x_gas)
    v_pms = np.array(np.log1p(x_pms))
    v_env = np.array((temp-20, humidity-50))
    v_prs = np.array((pressure-1000,))
    v_lgt = np.array((np.log(lux+.01),))

    # Display
    if i>0: # First reading bogus, skip it
        z_act = a_act.uz2(np.concatenate((v_lgt, v_env, v_gas)))
        z_env = a_env.uz2(np.concatenate((v_env, v_gas)))
        z_gpm = a_gpm.uz2(np.concatenate((v_prs, v_pms, v_env)))
        if (False):
            print(z_act, z_env, z_gpm)
            #print(np.linalg.eigvals(a_act.SS))
            #print(np.concatenate((v_lgt, v_env, v_gas)))
            #print(a_act.N, a_act.mean)
        # z with mean 1, sd \appr .6, so 2 is below two-sigma
        def barify(z): return int(min(255, 256*max(0, z-2)/4)) 
            
        display_everything({'temp': temp, 'hum' : humidity, 'pres' : pressure,
                            'pms010': x_pms[0], 'pms025': x_pms[1], 'pms100': x_pms[2] },
                           bars = ((barify(z_act), 0, 0),
                                   (0, barify(z_env), 0),
                                   (0, 0, barify(z_gpm))),
                           bg  = (0, 0, 0)) #(255*(proximity>0), 0, 0))

        # Logging
        x = np.array((time.time(), temp, pressure, humidity, lux, proximity) +
                         x_gas + x_pms + (z_act,), 'float')
        xs.append(x) 
        v_lngs.append(np.concatenate((v_gas, v_env, v_prs, v_pms, v_lgt)))

    if i % 60 == 0 and i>0: # Skip the round with no data
        xm = np.mean(xs, 0); xs = []
        v_lng = np.mean(v_lngs, 0); v_lngs = []
        z_lng = a_lng.uz2(v_lng)
        print("%d %5.2f %6.2f %4.2f %8.2f %6.2f %7.3f %7.3f %7.3f %6.2f %6.2f %6.2f %5.2f %5.2f" %
                  (tuple(xm) + (z_lng,)))
        sys.stdout.flush()

    if i % 1800 == 0:
        pickle.dump((a_act, a_env, a_gpm, a_lng), open(INTEGRATORS_FILENAME, 'wb'))
        
    time.sleep(1.0)
                    
