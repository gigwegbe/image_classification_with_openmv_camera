# Edge Impulse - OpenMV Image Classification Example

import sensor, image, time, os, tf, lcd, ulab

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.set_framesize(sensor.LCD)
sensor.skip_frames(time=2000)          # Let the camera adjust.

#sensor.skip_frames()
lcd.init()

net = "mouse_quantized.tflite"
labels = [line.rstrip('\n') for line in open("labels.txt")]

clock = time.clock()
while(True):
    clock.tick()

    img = sensor.snapshot()
    # default settings just do one detection... change them to search the image...
    for obj in tf.classify(net, img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
        #print("**********\nPredictions at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
        img.draw_rectangle(obj.rect())
        # This combines the labels and confidence values into a list of tuples
        predictions_list = list(zip(labels, obj.output()))
        # Convert the output to array
        obj_arr = ulab.numpy.array(obj.output())
        # Get the position of the max. value
        obj_arg = ulab.numpy.argmax(obj_arr)
        # Get the label
        label_text = labels[obj_arg]
        # Draw text to the image
        img.draw_string(0,0,label_text)
        # Push image to lcd
        lcd.display(img)
