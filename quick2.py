import time, asyncio #, random
import numpy as np
import bme280
import RPi.GPIO as GPIO

LOG_FILENAME = "3.txt"

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

def bme_read(b):
    return time.time(), b.get_temperature(), b.get_humidity(), b.get_pressure()

async def bme_get():
    GPIO.setup(17, GPIO.OUT, initial=0)
    await asyncio.sleep(.1)
    GPIO.output(17, 1)
    b = bme280.BME280()
    bme_read(b)
    await asyncio.sleep(.1) # the combo of reading and sleep is needed to get first corrupt meas. out
    return b

async def measure(q_meas, interval=1.0):
    b = await bme_get()
    t_sync = time.time()
    while True:
        try:
            await q_meas.put(bme_read(b))
        except OSError as e: # was it dead?
            b = await bme_get()
        await asyncio.sleep(interval - (time.time() - t_sync) % interval)
        # if random.random() < -1: GPIO.output(17, 0) #kill it delirebately

async def avg(q_meas):
    logfile = open(LOG_FILENAME, "a")
    while True:
        obs = np.array([await q_meas.get() for i in range(600)])
        mean = np.mean(obs, 0)
        obs0 = obs - mean
        t0 = obs0[:,0] 
        slope = np.sum((obs0 * t0[:, np.newaxis]), 0) / np.sum((t0 * t0)[:, np.newaxis], 0)
        resid_std = np.std(obs0 - np.outer(t0, slope), 0)
        logfile.write('%d %6.2f %4.2f %4.2f    %6.2f %6.2f %6.2f   %6.2f %6.2f %6.2f %3d\n' %
                      (tuple(mean) + tuple(3600*slope[1:]) + tuple(1000*resid_std[1:]) + (len(obs),)))
        logfile.flush()
        
    
async def main():
    q_meas = asyncio.Queue()
    await asyncio.gather(measure(q_meas), avg(q_meas))

asyncio.run(main())
