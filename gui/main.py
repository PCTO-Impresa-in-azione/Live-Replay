from typing import cast
from pyguiStyle import load_def_style

from dearpygui.core import *
from dearpygui.simple import *
from dearpygui.demo import *
import cv2
import os, shutil

def SvuotaCache():
    for filename in os.listdir("./cache"):
        file_path = os.path.join("./cache", filename)                
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def LoadImage(sender, data):
    SvuotaCache()
    vid = cv2.VideoCapture(data[0] + "/" + data[1]) #ottengo il video dal percorso selezionato
    with window("Preview"): #creo una finesta pygui dove mostrare il video     
        add_drawing("Immagine (nome file)")
        i = 0 #usato per cache
        while(True):            
            try:
                re,img = vid.read() #ottengo un frame
                cv2.imwrite("./cache/{0}_{1}_cache.jpg".format(i, data[1]), img) #salvo il frame nella cache
                img = cv2.resize(img, None, fx=1.0, fy=1.0) #non usato (ridimensiona l`immagine)

                height, width, _ = img.shape #imposto la dimensione della finestra preview in base al video
                set_item_height("Preview", height + 20)
                set_item_height("Immagine (nome file)", height)
                set_item_width("Preview", width + 20)
                set_item_width("Immagine (nome file)", width)

                draw_image("Immagine (nome file)", "./cache/{0}_{1}_cache.jpg".format(i, data[1]), [0, 0], [width, height]) #carico il frame dalla cache
               
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if(i > 1):
                    os.remove("./cache/{0}_{1}_cache.jpg".format(i - 1, data[1])) #elimino il file inutile dalla cache
                i = i + 1
            except:
                SvuotaCache()
                vid.release()
                return
        SvuotaCache()
        vid.release()           

def BtnFileSelectClick():
    #TODO custom extensions
    open_file_dialog(LoadImage)

enable_docking(shift_only = False)
set_main_window_size(1920, 1080)
set_main_window_title("SportView")

add_additional_font("resources/fonts/Poppins/Poppins-Medium.ttf", size = 15.0)

load_def_style()

with window("Docking canvas", width = 1920, height = 1080, no_move = True, no_resize = True, no_background = True, no_title_bar=True, x_pos = 0, y_pos = 0, no_bring_to_front_on_focus = True):
    set_primary_window("Docking canvas", True)

with window("File"):
    add_button("Select a file", callback = BtnFileSelectClick)

start_dearpygui()
stop_dearpygui()