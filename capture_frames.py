import cv2
from pathlib import Path

video_path = 'data/IMG_3582.mov'
images_dirpath = Path(__file__).parent / 'images' / 'IMG_3582'  # Path(__file__).parent / 'images'
images_dirpath.mkdir(exist_ok=True)

cap = cv2.VideoCapture(video_path)
cnt = 0

while(True):
    ret, frame = cap.read()
    cv2.imshow('video_path', frame)
    if (cnt % 20) == 0:
        # if cv2.waitKey(1) & 0xFF == ord('y'):
        frame_filepath = images_dirpath / f'frame_{cnt}.png'
        cv2.imwrite(str(frame_filepath), frame)
        # cv2.destroyAllWindows()
        print(f"File written: {frame_filepath}")
        # break
    cnt += 1

cap.release()
# stream_server.stop()
