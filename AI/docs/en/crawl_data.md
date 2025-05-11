# YouTube Video Downloader by Topic

The script uses `yt_dlp` to download videos from YouTube playlists and categorize them into folders according to actions like `abuse`, `arrest`, `robbery`, etc.

## Requirements

* Python 3.10 or later
* `yt_dlp` library
* Cookies from browser to access restricted videos (if needed)

## Recommendations

* Conda is recommended for creating virtual environments:

```bash
conda create -n yt_downloader python=3.10
conda activate yt_downloader
pip install yt-dlp
```

* To download restricted videos or videos that require login, you need to use an extension to get cookies from your browser:

* Extension: [Get cookies.txt](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)
* After installing, go to YouTube, log in with your account, click on the extension to download the cookies file.
* Save the file (any name is fine, for example `cookies.txt`).

## Run the script

1. Place the cookies file in the same directory as the script, and update the following line in the Python code if the file name is different:

```python
"cookiefile": "your_cookies_filename.txt"
```

2. Then run:

```bash
python crawl_data.py
```

## Result

The video will be uploaded to the `crawled data` folder, each topic is a subfolder. The video is named in the format:

```
crawled_{topic_name}_{000001}.mp4
```

For example:

```
crawled data/
├— abuse/
│   └— crawled_abuse_000001.mp4
├— robbery/
│   └— crawled_robbery_000001.mp4
...

## Note

* Cookies are **machine specific**, so each person needs to create their own using their browser.

* If you encounter any errors, please check your playlist URL or update `yt_dlp` with the command:

```bash
pip install -U yt-dlp
```
## Organize Data

Collected YouTube videos are categorized into specific action-based folders such as:

- `abuse`
- `arrest`
- `robbery`
- `normal`
- `explosion`
- `fight`
- ...

Each playlist contains surveillance or incident footage matching the category. This helps structure the dataset for action recognition or anomaly detection.

Categorized YouTube playlists by action
![image alt](https://github.com/diligent-man/Video_Anomaly_Detection/blob/90e31539a09febf904ac2c6050d6b5269f6c47e4/data_youtube.png)
