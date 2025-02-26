import torch
import ffmpeg

from AI.src.utils.load_video import v2


def _find_video_stream(streams) -> str:
    __CODECS = [
        "h264", "mpeg4"
    ]

    for i, stream in enumerate(streams):
        if stream["codec_name"] in __CODECS:
            return str(i)

def main() -> None:
    fpath = "/home/trong/Downloads/Dataset/VAD/iitb/labeled/anomaly/bagexchange/000232.avi"
    probe_info = ffmpeg.probe(fpath)

    filters1 = {
        "fps": {"fps": 10, "round": "up"},
        "scale": {"w": 256, "h": 256, "sws_flags": "lanczos"},
        "crop": {"out_w": 224, "out_h": 224, "exact": 1, "keep_aspect": 1},
    }

    stream = ffmpeg.input(fpath)[(_find_video_stream(probe_info["streams"]))]
    for filter_name, kwargs in filters1.items():
        stream = stream.filter(filter_name, **kwargs)

    stream = stream.output("/home/trong/Downloads/with_fps.mp4", pix_fmt="rgb24", loglevel="quiet")
    stream = stream.overwrite_output()
    stream.run()
    #############################
    stream = ffmpeg.input(fpath)[(_find_video_stream(probe_info["streams"]))]
    for filter_name, kwargs in filters1.items():
        stream = stream.filter(filter_name, **kwargs)

    stream = stream.output("/home/trong/Downloads/without_fps.mp4", pix_fmt="rgb24", loglevel="quiet")
    stream = stream.overwrite_output()
    stream.run()

    with_fps = v2(fpath, device="cuda")
    without_fps = v2(fpath, device="cuda")

    print(without_fps.shape)
    print(with_fps.shape)


    return None

if __name__ == '__main__':
    main()