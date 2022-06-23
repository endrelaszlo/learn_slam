import cv2
from pathlib import Path

images_dirpath = Path(__file__).parent / 'images'
images_dirpath.mkdir(exist_ok=True)

# cap = cv2.VideoCapture(0)

from stream_server import StreamServer

stream_server = StreamServer()
cap = stream_server.start()

ret, frame = cap.read()

cnt = 0

while(True):
    ret, frame = cap.read()
    cv2.imshow('camera', frame)
    if cv2.waitKey() & 0xFF == ord('y'):
        frame_filepath = images_dirpath / f'frame_{cnt}.png'
        cv2.imwrite(str(frame_filepath), frame)
        # cv2.destroyAllWindows()
        print(f"File written: {frame_filepath}")
        cnt += 1


cap.release()
stream_server.stop()
