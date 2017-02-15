import sys
import random
import time
from multiprocessing import Process, Queue
from PIL import Image

def worker(q):
    from tkinter import Tk, Label
    from PIL import ImageTk

    def escape():
        sys.exit(0)
    
    def task():
        try:
            im, score, label = q.get(block=False)
            photo = ImageTk.PhotoImage(im)
            image_label.config(image=photo)
            image_label.image = photo
            image_label.pack(side="bottom", fill="both", expand="yes")
            text = "Score for %s: %f" % (label, score)
            score_label.config(text=text)
        except:
            pass

        root.after(100, task)

    root = Tk()
    root.wm_title('drawobjects-v0')
    image_label = Label(root)
    score_label = Label(root)
    score_label.pack()
    root.after(100, task)
    root.bind('<Control-slash>', escape)
    root.mainloop()

class GUI(object):
    def __init__(self):
        self.queue = Queue()
        self.process = Process(target=worker, args=(self.queue,))

    def start(self):
        self.process.start()

    def update(self, *args):
        self.queue.put(args)

def random_image():
    width = 200
    height = 200

    im = Image.new("1", (width, height), (255,255,255))
    pixel = im.load()

    for x in range(200):
        for y in range(200):
            if random.random() < 0.01:
                pixel[x,y] = 1
            else:
                pixel[x,y] = 0

    return im

def main():
    gui = GUI()
    gui.start()

    while True:
        gui.update(random_image(), 0.0, "test")
        time.sleep(1)

if __name__ == '__main__':
    main()

