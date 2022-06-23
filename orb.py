import numpy as np
import cv2
# import cv2.cv as cv
from pytube import YouTube
import matplotlib.pyplot as plt


def capture_youtube():
    #
    # Download YouTube clip
    #
    yt_clip_url = 'https://www.youtube.com/watch?v=KrvXLbjTgMQ'
    # yt_clip_url = 'https://www.youtube.com/watch?v=kSVqPAN1nCM'
    # yt_clip_url = 'https://www.youtube.com/watch?v=ulGchsqJ89g'
    filename = 'video.mp4'
    fps = 30
    yt = YouTube(yt_clip_url)
    yt_stream = yt.streams.filter(fps=fps, resolution='1080p', mime_type='video/mp4').desc().first()

    #
    # Open stream based on file or url
    #
    use_url = False
    if use_url:
        cap = cv2.VideoCapture(yt_stream.url)
    else:
        yt_stream.download(filename=filename)
        cap = cv2.VideoCapture(filename)

    assert np.isclose(fps, cap.get(cv2.CAP_PROP_FPS), atol=1.0)
    frame_count = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
    frame_start = 1000
    assert frame_start < frame_count
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    # cap.set()

    # check if url was opened
    if not cap.isOpened():
        print('video not opened')
        exit(-1)

    return cap


# def capture_camera():
#     """iPhone 12 mini photo camera"""
#     mtx = np.array([[1.40623850e+03, 0.00000000e+00, 9.67314690e+02],
#                     [0.00000000e+00, 1.40341000e+03, 5.48343323e+02],
#                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
#
#     newcameramtx = np.array([[1.41210034e+03, 0.00000000e+00, 9.73388794e+02],
#                              [0.00000000e+00, 1.40264819e+03, 5.47775516e+02],
#                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
#
#     dist = np.array([[ 0.07299342, -0.03285733, -0.00026655, 0.00237489, -0.12026614]])
#     cap = cv2.VideoCapture(0)
#     return cap, mtx, newcameramtx, dist


def capture_iphone_video_camera_recording_calib():
    """iPhone 12 mini video camera"""
    mtx = np.array([[1.66859300e+03, 0.00000000e+00, 9.60517268e+02],
                    [0.00000000e+00, 1.66970844e+03, 5.38656265e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    newcameramtx = np.array([[1.70475659e+03, 0.00000000e+00, 9.60759466e+02],
                             [0.00000000e+00, 1.68863672e+03, 5.38793790e+02],
                             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    dist = np.array([[1.99596860e-01, -9.53049505e-01, 1.22922644e-03, 5.24131736e-04, 1.71807240e+00]])
    video_path = 'data/IMG_3584.mov'
    cap = cv2.VideoCapture(video_path)
    return cap, mtx, newcameramtx, dist


def capture_iphone_stream_camera_calib():
    """iPhone 12"""
    mtx = np.array([[1.39842094e+03, 0.00000000e+00, 9.55587395e+02],
                    [0.00000000e+00, 1.39825911e+03, 5.07148491e+02],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    newcameramtx = np.array([[1.44201953e+03, 0.00000000e+00, 9.55543827e+02],
                             [0.00000000e+00, 1.41559937e+03, 5.07167403e+02],
                             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    dist = np.array([[ 1.74592360e-01, -7.35454528e-01, 3.05461458e-04, -1.05571253e-04, 1.13439385e+00]])
    from stream_server import StreamServer
    stream_server = StreamServer()
    cap = stream_server.start()
    return cap, mtx, newcameramtx, dist, stream_server


def capture_stream():
    from stream_server import StreamServer
    stream_server = StreamServer()
    cap = stream_server.start()
    return cap, stream_server


if __name__ == '__main__':

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # cap = capture_youtube()
    # cap = capture_camera()
    cap, mtx, newcameramtx, dist = capture_iphone_video_camera_recording_calib()
    # cap, stream_server = capture_stream()

    # cap, mtx, newcameramtx, dist, stream_server = capture_stream_camera_calib()
    # cap.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    feature_extractor = cv2.ORB_create(nfeatures=200)
    # feature_extractor = cv2.SIFT_create(nfeatures=200)
    # feature_matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False)
    # FLANN parameters
    FLANN_INDEX_LINEAR = 0
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_KMEANS = 2
    FLANN_INDEX_COMPOSITE = 3
    FLANN_INDEX_KDTREE_SINGLE = 4
    FLANN_INDEX_HIERARCHICAL = 5
    FLANN_INDEX_LSH = 6
    index_params = {'algorithm': FLANN_INDEX_HIERARCHICAL}  # , 'trees': 2}
    search_params = {'checks': 50}   # or pass empty dictionary
    feature_matcher = cv2.FlannBasedMatcher(index_params, search_params)

    prev_frame = None
    prev_kps, prev_descs = None, None

    skip_nth_frame = 0
    skip_counter = skip_nth_frame

    def draw_epipolar_lines(img1, img2, lines, pts1, pts2):
        """img1 - image on witch we draw the epilines for the points in img2
           lines - corresponding epilines"""
        img1 = img1.copy()
        img2 = img2.copy()
        ii32 = np.iinfo(np.int32)
        cols = img1.shape[1]
        # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        for line, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(np.int32, [0, -line[2]/line[1]])
            # x1, y1 = map(int, [cols, -(line[2] + line[0]*cols/line[1])])
            x1, y1 = map(np.int32, [cols, -(line[2] + line[0]*cols)/line[1]])
            # y0, y1 = map(lambda val: min(max(val, ii32.min), ii32.max), [y0, y1])
            # print(x0, y0, x1, y1)
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
            img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
        return img1, img2

    xdata = list()
    ydata = list()
    zdata = list()
    colors = list()

    cnt = 0
    while True:
        # read frame
        ret, cur_frame = cap.read()

        # check if frame is empty
        if not ret:
            break

        if skip_counter == 0:
            skip_counter = skip_nth_frame
        else:
            skip_counter -= 1
            continue

        cur_kps, cur_descs = feature_extractor.detectAndCompute(image=cur_frame, mask=None)
        cur_descs = np.float32(cur_descs)
        # print(f'cur_kps: {len(cur_kps)}')

        cur_frame_kps = cur_frame
        # for kp in cur_kps:
        #     cur_frame_kps = cv2.circle(img=cur_frame_kps, center=(int(kp.pt[0]), int(kp.pt[1])), radius=5, color=(57, 255, 20), thickness=3)

        if prev_descs is not None:
            # matches = feature_matcher.match(queryDescriptors=cur_descs, trainDescriptors=prev_descs)
            matches = feature_matcher.knnMatch(queryDescriptors=cur_descs, trainDescriptors=prev_descs, k=2)

            # matches = sorted(matches, key=lambda x: x.distance)[:100]
            # for m in matches:
            #     cur_kp = cur_kps[m.queryIdx]
            #     prev_kp = prev_kps[m.trainIdx]
            #     cur_kp = (int(cur_kp.pt[0]), int(cur_kp.pt[1]))
            #     prev_kp = (int(prev_kp.pt[0]), int(prev_kp.pt[1]))
            #     cur_frame_kps = cv2.line(img=cur_frame_kps, pt1=cur_kp, pt2=prev_kp, color=(57, 255, 20), thickness=3)
            # cv2.imshow('frame', cur_frame_kps)

            # # matches = sorted(matches, key=lambda x: x[0].distance)[:100]
            # for m in matches:
            #     m = m[0]
            #     cur_kp = cur_kps[m.queryIdx]
            #     prev_kp = prev_kps[m.trainIdx]
            #     cur_kp = (int(cur_kp.pt[0]), int(cur_kp.pt[1]))
            #     prev_kp = (int(prev_kp.pt[0]), int(prev_kp.pt[1]))
            #     cur_frame_kps = cv2.line(img=cur_frame_kps, pt1=cur_kp, pt2=prev_kp, color=(57, 255, 20), thickness=3)
            # cv2.imshow('frame', cur_frame_kps)


            # # Need to draw only good matches, so create a mask
            # matchesMask = [[0, 0] for i in range(len(matches))]
            #
            # # ratio test as per Lowe's paper
            # for i, (m, n) in enumerate(matches):
            #     if m.distance < 0.7 * n.distance:
            #         matchesMask[i] = [1, 0]
            # draw_params = dict(matchColor=(0, 255, 0),
            #                    singlePointColor=(255, 0, 0),
            #                    matchesMask=matchesMask,
            #                    flags=0)
            #
            # cur_frame_kps = cv2.drawMatchesKnn(cur_frame, cur_kps, prev_frame, prev_kps, matches, None, **draw_params)
            # cv2.imshow('frame', cur_frame_kps)

            # matching_results = cv2.drawMatches(cur_frame, cur_kps, prev_frame, prev_kps, matches, None, matchesMask=matchesMask,
            #                    flags=1)
            # cv2.imshow('frame', matching_results)

            good = []
            pts1 = []
            pts2 = []

            # ratio test as per Lowe's paper
            for i, (m, n) in enumerate(matches):
                # if m.distance < 0.8 * n.distance:
                good.append(m)
                pts1.append(cur_kps[m.queryIdx].pt)
                pts2.append(prev_kps[m.trainIdx].pt)

            print(len(pts1))
            pts1 = np.int32(pts1)
            pts2 = np.int32(pts2)
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

            # We select only inlier points
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]

            # Find epilines corresponding to points in right image (second image) and
            # drawing its lines on left image
            lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
            lines1 = lines1.reshape(-1, 3)
            img5, img6 = draw_epipolar_lines(cur_frame, prev_frame, lines1, pts1, pts2)
            cv2.imshow('frame', img5)

            # # Find epilines corresponding to points in left image (first image) and
            # # drawing its lines on right image
            # lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
            # lines2 = lines2.reshape(-1, 3)
            # img3, img4 = draw_epipolar_lines(prev_frame, cur_frame, lines2, pts2, pts1)
            #
            # img_to_show = np.concatenate([img5, img3], axis=1)
            # cv2.imshow('frame', img_to_show)

            # E = mtx.T * F * mtx
            E, mask = cv2.findEssentialMat(pts1, pts2, mtx, cv2.FM_RANSAC)
            # findEssentialMat
            ret, R, t, mask = cv2.recoverPose(E, pts1, pts2, mtx, mask=mask)
            # print(t)
            t = t.squeeze()
            xdata.append(xdata[-1]+t[0] if xdata else t[0])
            ydata.append(ydata[-1]+t[1] if ydata else t[1])
            zdata.append(zdata[-1]+t[2] if zdata else t[2])
            colors.append(cnt)
            cnt += 1


        # cv2.imshow('frame', cv2.circle(img=frame, center=(447, 63), radius=63, color=(0, 0, 255))) #, thickness=-1))

        # display frame
        # cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('g'):
            ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');
            plt.show()
            break

        prev_frame = cur_frame
        prev_kps, prev_descs = cur_kps, cur_descs

    # release VideoCapture
    cap.release()
    cv2.destroyAllWindows()
    # stream_server.stop()


ax.scatter3D(xdata, ydata, zdata, c=colors, cmap='Greens');
plt.show()
