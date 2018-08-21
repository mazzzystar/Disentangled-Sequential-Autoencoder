import base64
import time
from selenium import webdriver
from selenium.webdriver.support.ui import Select

driver = webdriver.Firefox()
driver.get("http://gaurav.munjal.us/Universal-LPC-Spritesheet-Character-Generator/")
driver.maximize_window()
#bodies = ['body-darkelf','body-dark2','body-dark']

#for body in bodies:
#    print('Working on ' + body)
#    element = driver.find_element_by_id(body)
#    driver.execute_script("return arguments[0].click();",element)
#    canvas = driver.find_element_by_id('spritesheet')
#    canvas_base64 = driver.execute_script("return arguments[0].toDataURL('image/png').substring(21);",canvas)
#    canvas_png = base64.b64decode(canvas_base64)
#    with open(body+".png","wb") as f:
#        f.write(canvas_png)

bodies = ['light','dark','dark2','darkelf','darkelf2','tanned','tanned2']
shirts = ['longsleeve_brown','longsleeve_teal','longsleeve_maroon','longsleeve_white']
hairstyles = ['green','blue','pink','raven','white','dark_blonde']
pants = ['magenta','red','teal','white','robe_skirt']
count = 0
for body in bodies:
    driver.execute_script("return arguments[0].click();",driver.find_element_by_id('body-'+body))
    time.sleep(1)
    for shirt in shirts:
        driver.execute_script("return arguments[0].click();",driver.find_element_by_id('clothes-'+shirt))
        time.sleep(1)
        for pant in pants:
            if pant=='robe_skirt':
                driver.execute_script("return arguments[0].click();",driver.find_element_by_id('legs-'+pant))
            else:
                driver.execute_script("return arguments[0].click();",driver.find_element_by_id('legs-pants_'+pant))
            time.sleep(1)
            for hair in hairstyles:
                driver.execute_script("return arguments[0].click();",driver.find_element_by_id('hair-plain_'+hair))
                time.sleep(1)
                name = body+"_"+shirt+"_"+pant+"_"+hair
                count = count+1 
                print("Creating character: "  + "'" + name)
                print("Characters so far : %d" % count)
                canvas = driver.find_element_by_id('spritesheet')
                canvas_base64 = driver.execute_script("return arguments[0].toDataURL('image/png').substring(21);",canvas)
                canvas_png = base64.b64decode(canvas_base64)
                with open(str(count-1) + ".png","wb") as f:
                    f.write(canvas_png)


