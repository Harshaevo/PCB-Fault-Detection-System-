import cv2
from tkinter import filedialog
import uuid



# Function for opening the
# file explorer window
def browseFiles():
    filename = filedialog.askopenfilename(initialdir="E:\Software Project\PCB Fault\FaultImages",
                                          title="Select a File",
                                          filetypes=(("CameraImages",
                                                      "*.jpg*"),
                                                     ("all files",
                                                      "*.*")))
    return filename


def camera():
    # define a video capture object
    vid= cv2.VideoCapture(0)


    while True:

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        # Display the resulting frame
        cv2.imshow('Live Camera', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            filename = f"CameraImages\Live Camera.jpg"
            cv2.imwrite(filename, frame)
            # After the loop release the cap object
            vid.release()
            # Destroy all the windows
            cv2.destroyAllWindows()
            return filename

def savefile(edge):
    filename = filedialog.asksaveasfile(mode='w', defaultextension=".jpg", initialdir = "/",title = "Select file",filetypes = (('JPEG', ('*.jpg','*.jpeg','*.jpe','*.jfif')),('PNG', '*.png'),('BMP', ('*.bmp','*.jdib')),('GIF', '*.gif')))
    if not filename:
        return
    edge.save(filename)