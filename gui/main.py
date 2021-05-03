from typing import cast
from pyguiStyle import load_def_style

from dearpygui.core import *
from dearpygui.simple import *
from dearpygui.demo import *
from dearpygui import core
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
set_main_window_size(1080, 720)
set_main_window_title("SportView")

add_additional_font("resources/fonts/Poppins/Poppins-Medium.ttf", size = 15.0)

load_def_style()

with window("Docking canvas", width = 1080, height = 720, no_move = True, no_resize = True, no_background = True, no_title_bar=True, x_pos = 0, y_pos = 0, no_bring_to_front_on_focus = True):
    set_primary_window("Docking canvas", True)

with window("homepage", width=720, height=600):
    core.add_image(name="esempio", value="gui/resources/images/eses.png", height = 416, width = 958)
    add_button("Seleziona un file", callback = BtnFileSelectClick)
    add_button("Stats giocatore")
    add_button("Gol più bello")
    add_button("Parata migliore")

with window("Gol migliore", width=720, height=600):
    core.add_image(name="esempio1", value="gui/resources/images/gol.png", height = 328, width=552)
    add_text("Giocatore che ha segnato: Alex Sandro")
    add_text("Squadra del giocatore: Juventus")
    add_text("Come ha segnato: Colpo di testa")

with window("Stats squadra", width=720, height=600):
    add_text("Squadra 1")
    add_text("Passaggi effettuati: 87")
    add_text("Tiri effettuati: 12")
    add_text("Goal: 2")
    add_text("Calci d'angolo: 3")
    add_text("Fuori gioco: 2")
    add_text("Cartellini gialli; 1")
    add_text("Cartellini rossi: 0")
    add_text("Falli fatti: 0")
    add_text("Possesso palla: 65%")
    add_text("Tiri in porta: 7")
    add_text(" ")
    add_text("Squadra 2")
    add_text("Passaggi effettuati: 76")
    add_text("Tiri effettuati: 8")
    add_text("Goal: 1")
    add_text("Calci d'angolo: 2")
    add_text("Fuori gioco: 3")
    add_text("Cartellini gialli: 1")
    add_text("Cartellini rossi: 1")
    add_text("Falli fatti: 1")
    add_text("Possesso palla: 45%")
    add_text("Tiri in porta: 4")

with window("Parata", width=720, height=600):
    core.add_text("Miglior parata")
    core.add_image(name="esempio2", value="gui/resources/images/parata.jpeg", height = 328, width=552)
    core.add_text("Nome portiere: Gianluigi Buffon")
    core.add_text("Squadra del portiere: Juventus")
    core.add_text("A chi è stato parato il goal: Christian Eriksen")

with window("Giocatori", width=720, height=600):
    core.add_text("Giocatore 1")
    core.add_image(name="1", value="gui/resources/images/429.png", height = 200, width=200)
    core.add_text("Minuti Giocati: 65")
    core.add_text("Passaggi eseguiti: 47")
    core.add_text("Tiri in porta: 1")
    core.add_text("Gol: 1")
    core.add_text("Giocatore 2")
    core.add_image(name="2", value="gui/resources/images/20801.png", height = 200, width=200)
    core.add_text("Minuti Giocati: 57")
    core.add_text("Passaggi eseguiti: 32")
    core.add_text("Tiri in porta: 3")
    core.add_text("Gol: 0")
    core.add_text("Giocatore 3")
    core.add_image(name="3", value="gui/resources/images/193082.png", height = 200, width=200)
    core.add_text("Minuti Giocati: 43")
    core.add_text("Passaggi eseguiti: 23")
    core.add_text("Tiri in porta: 2")
    core.add_text("Gol: 1")
    # core.add_text("Giocatore 4")
    # core.add_image(name="4", value="gui/resources/images/kek.jpg", height = 200, width=200)
    # core.add_text("Giocatore 5")
    # core.add_image(name="5", value="gui/resources/images/kek.jpg", height = 200, width=200)
    # core.add_text("Giocatore 6")
    # core.add_image(name="6", value="gui/resources/images/kek.jpg", height = 200, width=200)
    # core.add_text("Giocatore 7")
    # core.add_image(name="7", value="gui/resources/images/kek.jpg", height = 200, width=200)
    # core.add_text("Giocatore 8")
    # core.add_image(name="8", value="gui/resources/images/kek.jpg", height = 200, width=200)
    # core.add_text("Giocatore 9")
    # core.add_image(name="9", value="gui/resources/images/kek.jpg", height = 200, width=200)
    # core.add_text("Giocatore 10")
    # core.add_image(name="10", value="gui/resources/images/kek.jpg", height = 200, width=200)
    # core.add_text("Giocatore 11")
    # core.add_image(name="11", value="gui/resources/images/kek.jpg", height = 200, width=200)

start_dearpygui()
stop_dearpygui()