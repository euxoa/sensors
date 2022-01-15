#!/usr/bin/env python3
import time, colorsys, sys, os, pickle, itertools, asyncio
import numpy as np

import ST7735
from bme280 import BME280
from pms5003 import PMS5003, ReadTimeoutError as pmsReadTimeoutError, SerialTimeoutError
from enviroplus import gas
from ltr559 import LTR559

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from fonts.ttf import RobotoMedium as UserFont

INTEGRATORS_FILENAME = "integrators.pyobj"
LOG_FILENAME = "3.txt"

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
    def __call__(self, v):
        self.update(v)
        return self.z2(v)

async def display(q_display):
    # Create display instance
    st7735 = ST7735.ST7735(port=0, cs=1, dc=9, backlight=12, rotation=270,
                           spi_speed_hz=10000000)
    st7735.begin()
    WIDTH, HEIGHT = st7735.width, st7735.height
    img = Image.new('RGB', (WIDTH, HEIGHT), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(UserFont, 18)
    x_offset, y_offset = 2, 2
    limits = {'temp': [16, 19, 21.5, 24, 28],
              'hum': [18, 28, 40, 55, 70],
              'pres' : [970, 985, 1000, 1018, 1035],
              'pms010': [0, 1, 2, 10, 25],
              'pms025': [0, 1, 2, 10, 25],
              'pms100': [0, 1, 2, 10, 25] }
    palette = np.array([(220, 0, 220), (0, 50, 255), (50, 200, 50), (200, 200, 0), (255, 0, 0)])

    while True:
        content = await q_display.get()
        values = content['variables']
        bg = content.get('bg', (0, 0, 0))
        bars = content.get('bars')
        draw.rectangle((0, 0, WIDTH, HEIGHT), bg)
        if bars is not None:
            tri = HEIGHT//3 # FIXME: misses the quotient
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
            rgb = tuple([int(np.interp(value, lim, palette[:, i])) for i in (0, 1, 2)])
            draw.text((x, y), message, font=font, fill=rgb)
        st7735.display(img)

async def measure(q_measure, interval = 1.0):
    ltr559 = LTR559() # light
    ltr559.set_light_options(gain=96)
    ltr559.set_light_integration_time_ms(400)
    bme280 = BME280() # temp etc
    pms5003 = PMS5003(); time.sleep(1.0) # particulates; seems somewhat delicate
    t_sync = time.time()
    while True:
        try:
            await q_measure.put((
                   time.time(),
                   ltr559.get_proximity(),
                   ltr559.get_lux(),
                   bme280.get_temperature(),
                   bme280.get_humidity(),
                   bme280.get_pressure(),
                   pms5003.read(), # pms5003.ChecksumMismatchError
                   gas.read_all()))
        except pms5003.ChecksumMismatchError:
            print("Checksum error from PMS5003 at ", time.asctime())
        await asyncio.sleep(interval - (time.time() - t_sync) % interval)

if os.path.isfile(INTEGRATORS_FILENAME):
        print("Reading integrators.")
        a_act, a_env, a_gpm, a_lng = pickle.load(open(INTEGRATORS_FILENAME, 'rb'))
else:
        a_act = Anom(6, .001) # "nearby activity" half time about 10 minutes (sample per sec)
        a_env = Anom(5, .0002) # temp, hum and gas; an hour
        a_gpm = Anom(6, .00005) # slow-moving pres, env and pms; five hours
        a_lng = Anom(10, .0001) # half time about five days (sample per minute)

async def avg_logger(q_logger, a_lng):
    logfile = open(LOG_FILENAME, "a")
    while True:
        xs, v_lngs = [], []
        for i in range(60):
            x, v = await q_logger.get()
            xs.append(x); v_lngs.append(v)
        xm = np.mean(xs, 0)
        v_lng = np.mean(v_lngs, 0)
        z_lng = a_lng(v_lng)
        logfile.write("%d %5.2f %6.2f %4.2f %8.2f %6.2f %7.3f %7.3f %7.3f %6.2f %6.2f %6.2f %5.2f %5.2f\n" %
                          (tuple(xm) + (z_lng,)))
        logfile.flush()                

async def integrator_saver(integrators):
    while True:
        await asyncio.sleep(5 * 60)
        pickle.dump(integrators, open(INTEGRATORS_FILENAME, 'wb'))
        
async def loop0(q_measure, q_display, q_logger):
    for i in itertools.count():
        t0, proximity, lux, temp, humidity, pressure, pms_obj, gas_obj = await q_measure.get()
        if i<2: continue # First reading(s) bogus, skip them. i<1 likely enough.
        
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
        z_act = a_act(np.concatenate((v_lgt, v_env, v_gas)))
        z_env = a_env(np.concatenate((v_env, v_gas)))
        z_gpm = a_gpm(np.concatenate((v_prs, v_pms, v_env)))
        
        # z with mean 1, sd \appr .6, so 2 is below two-sigma
        # properly, this shoudl still take df in, do a chisq thing
        def barify(z): return int(min(255, 256*max(0, z-2)/4)) 
        await q_display.put({'variables':
                                 {'temp': temp, 'hum' : humidity, 'pres' : pressure,
                                  'pms010': x_pms[0], 'pms025': x_pms[1], 'pms100': x_pms[2] },
                             'bars' : ((barify(z_act), 0, 0),
                                       (0, barify(z_env), 0),
                                       (0, 0, barify(z_gpm))),
                             'bg' : (0, 0, 0)}) #(255*(proximity>0), 0, 0))

        # Logging
        x = (t0, temp, pressure, humidity, lux, proximity) + x_gas + x_pms + (z_act,)
        v = np.concatenate((v_gas, v_env, v_prs, v_pms, v_lgt))
        await q_logger.put((np.array(x), v))

async def main():
    q_logger = asyncio.Queue()
    q_measure = asyncio.Queue()
    q_display = asyncio.Queue()
    await asyncio.gather(loop0(q_measure, q_display, q_logger),
                         measure(q_measure), 
                         avg_logger(q_logger, a_lng),
                         display(q_display),
                         integrator_saver((a_act, a_env, a_gpm, a_lng)))

asyncio.run(main())

# TODO
# - list comprehensions -> generators, in many places
# - the format string in logging, cut in half
# - shorthand for npcomc and nparray
# - the chisq df thing, requires scipy...
# - Integrators to local, they are in a funny place now.
# - logger to 60 seconds instead of rounds
# - take intervals and file names to parameters
# - PMS error
# - log file
#      - header
#      - a bit of rotation maybe, with automatic names?
#      - %#7.3g format?
# - a small web server
# - logging module for prints
# - I/O to async: logfile and pickle write?
# - exception handling, mainly KILL and C-c
# - integrators to a shelve?
