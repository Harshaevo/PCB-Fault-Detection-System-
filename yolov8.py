from ultralytics import YOLO
import cv2

def main(INPUT_IMAGE_URL):
    model = YOLO("best.pt")
    # from ndarray
    im2 = cv2.imread(INPUT_IMAGE_URL)
    results = model.predict(source=im2)
    names=results[0].names
    print(names)
    res_plotted = results[0].plot()
    filename = f"FaultImages\Classified.jpg"
    cv2.imwrite(filename, res_plotted)
    return names,filename