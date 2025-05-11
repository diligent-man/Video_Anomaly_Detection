# YouTube Video Downloader theo Chủ Đề

Script dùng `yt_dlp` để tải video từ các playlist YouTube và phân loại vào thư mục theo từng hành động như `abuse`, `arrest`, `robbery`, v.v.

## Yêu cầu

* Python 3.10 trở lên
* Thư viện `yt_dlp`
* File cookies từ trình duyệt để truy cập các video có giới hạn (nếu cần)

## Khuyến nghị

* Nên dùng Conda để tạo môi trường ảo:

```bash
conda create -n yt_downloader python=3.10
conda activate yt_downloader
pip install yt-dlp
```

* Để tải được video bị giới hạn hoặc cần đăng nhập, cần dùng extension để lấy cookies từ trình duyệt:

  * Extension: [Get cookies.txt](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)
  * Sau khi cài, vào YouTube, đăng nhập bằng tài khoản của bạn, nhấn vào extension để tải file cookies.
  * Lưu file lại (tên gì cũng được, ví dụ `cookies.txt`).

## Chạy script

1. Đặt file cookies trong cùng thư mục với script, và cập nhật dòng sau trong mã Python nếu tên file khác:

```python
"cookiefile": "tên_file_cookies_của_bạn.txt"
```

2. Sau đó chạy:

```bash
python crawl_data.py
```

## Kết quả

Video sẽ được tải vào thư mục `crawled data`, mỗi chủ đề là một thư mục con. Video được đặt tên theo định dạng:

```
crawled_{ten_chu_de}_{000001}.mp4
```

Ví dụ:

```
crawled data/
├— abuse/
│   └— crawled_abuse_000001.mp4
├— robbery/
│   └— crawled_robbery_000001.mp4
...

## Ghi chú

* File cookies là **tùy theo máy**, nên mỗi người cần tự tạo bằng trình duyệt của mình.
* Nếu gặp lỗi, hãy kiểm tra lại URL playlist hoặc cập nhật `yt_dlp` bằng lệnh:

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

