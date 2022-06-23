import cv2
import ffmpeg
import time
# import threading
import multiprocessing
from urllib.parse import urlunparse, ParseResult
import threading
import queue

# bufferless VideoCapture
class VideoCapture:

    def __init__(self, url_rtmp_cast):
        self.url_rtmp_cast = url_rtmp_cast
        self.cap = cv2.VideoCapture(self.url_rtmp_cast)
        self.q = queue.Queue()
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()
        self.ts_read_last = time.time_ns()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            # ret = self.cap.grab()
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            # ret, frame = self.cap.retrieve()
            self.q.put(frame)

    # # grab frames as soon as they are available
    # def _reader(self):
    #     while True:
    #         ret = self.cap.grab()
    #         # if not ret:
    #         #     print("Buffer empty!")
    #             # break

    def isOpened(self):
        return self.cap.isOpened()

    # retrieve latest frame
    def read(self):
        # frames_dropped = 0
        # while self.cap.grab():
        #     frames_dropped += 1
            # ret = self.cap.grab()
        # ret, frame = self.cap.read()
        # ret, frame = self.cap.retrieve()
        frame = self.q.get()
        ts_read_current = time.time_ns()
        timespan = ts_read_current - self.ts_read_last
        fps_prop = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Frame read rate: {1.0/(timespan*1e-9)}, set FPS={fps_prop}")
        self.ts_read_last = ts_read_current
        return True, frame

RTMP_PROTOCOL_VARIANT = 'rtmp'
RTMP_SERVER_IP = '192.168.1.109'
RTMP_APPLICATION = 'live'
RTMP_FORMAT = 'flv'
RTMP_STREAM = 'stream'
RTMP_SERVER_PORT_SINK = 1935
RTMP_SERVER_PORT_BROADCAST = 1936


def make_rtmp_url(sink_or_broadcast: str):
    scheme = RTMP_PROTOCOL_VARIANT
    if sink_or_broadcast == 'sink':
        netloc = f'{RTMP_SERVER_IP}:{RTMP_SERVER_PORT_SINK}'
    elif sink_or_broadcast == 'broadcast':
        netloc = f'{RTMP_SERVER_IP}:{RTMP_SERVER_PORT_BROADCAST}'
    path = f'{RTMP_APPLICATION}/{RTMP_STREAM}'
    parse_result = ParseResult(scheme, netloc, path, None, None, None)
    url = urlunparse(parse_result)
    return url


def run_stream_server_ffmpeg(url_sink: str, url_broadcast: str):
    stream = ffmpeg.input(url_sink, format=RTMP_FORMAT, listen=1)
    stream = stream.output(url_broadcast, format=RTMP_FORMAT, listen=1, codec="copy")
    stream.run()


class StreamServer:
    def __init__(self):
        self.url_rtmp_sink = make_rtmp_url(sink_or_broadcast='sink')
        self.url_rtmp_cast = make_rtmp_url(sink_or_broadcast='broadcast')

    def start(self):
        # self.proc = multiprocessing.Process(target=run_stream_server_ffmpeg, args=(self.url_rtmp_sink, self.url_rtmp_cast))
        self.proc = threading.Thread(target=run_stream_server_ffmpeg,
                                     args=(self.url_rtmp_sink, self.url_rtmp_cast))
        self.proc.daemon = True
        self.proc.start()

        # self.vcap = cv2.VideoCapture(self.url_rtmp_cast)
        self.cap = VideoCapture(self.url_rtmp_cast)
        while not self.cap.isOpened():
            print("Waiting for video stream source... ")
            time.sleep(1)
            # self.cap = cv2.VideoCapture(self.url_rtmp_cast)
            self.cap = VideoCapture(self.url_rtmp_cast)

        return self.cap

    def stop(self):
        self.proc.join()


if __name__ == '__main__':
    stream_server = StreamServer()
    cap = stream_server.start()
    ret, frame = cap.read()
    while True:
    # while ret and (frame is not None):
        ret, frame = cap.read()
        if frame is not None:
            cv2.imshow('VIDEO', frame)
        cv2.waitKey(1)

    stream_server.stop()
    cap.release()
    cv2.destroyAllWindows()
