import yt_dlp
import os


def main() -> None:
    video_lst = {
        # "https://www.youtube.com/watch?v=7GvH61Lab3g&list=PL3HxEb3s-wfbASMMkkr_3DcuLZCmhwqTn&pp=gAQB": "abuse",
        # "https://www.youtube.com/watch?v=g0kcA8SBkic&list=PL3HxEb3s-wfZGKtog930Sg7yaA-TnJt5a&pp=gAQB": "arrest",
        # "https://www.youtube.com/watch?v=9N3MHYoHTb4&list=PL3HxEb3s-wfblLts-IyAOc6T5wHr-L5tB&pp=gAQB": "arson",
        # "https://www.youtube.com/watch?v=u-VcErKk6Qs&list=PL3HxEb3s-wfaN8V_Bpg_SP9G1WptxJp35&pp=gAQB": "assault",
        # "https://www.youtube.com/watch?v=F5OVf8wkgQo&list=PL3HxEb3s-wfZX0lO6-PoRW7Oa05jeECEU&pp=gAQB": "burglary",
        # "https://www.youtube.com/watch?v=XxtJhKJh44I&list=PL3HxEb3s-wfYmsDZgHvbQZ0B0GvxbtjDh&pp=gAQB": "carry weapon",
        # "https://www.youtube.com/watch?v=dy8v4YI1C6s&list=PL3HxEb3s-wfbYVSOqXFuiDq8h_5RAyGyr&pp=gAQB": "chasing",
        # "https://www.youtube.com/watch?v=k8IWLKBbQeI&list=PL3HxEb3s-wfZ6ScaEVIgC36PtJnLrshod&pp=gAQB": "explosion",
        # "https://www.youtube.com/watch?v=vITeVsOr6O4&list=PL3HxEb3s-wfZCCpSs4n20ZTFxscWAXwO2&pp=gAQB": "fight",
        # "https://www.youtube.com/watch?v=ZjNjfuv20QY&list=PL3HxEb3s-wfZ3ZoakyLbuVOqesBlyfy23&pp=gAQB":"protest",
        # "https://www.youtube.com/watch?v=XY-f7KyIhs8&list=PL3HxEb3s-wfZ23UQErf5co7q4X53qFAdx&pp=gAQB":"road accidents",
        # "https://www.youtube.com/watch?v=KTDen9ooazo&list=PL3HxEb3s-wfbzY6eWisuKbfDnXDR0Buqy&pp=gAQB":"robbery",
        # "https://www.youtube.com/watch?v=R3AiPN0LBcY&list=PL3HxEb3s-wfZMGfHYclSfAp7_fpApQGPx&pp=gAQB":"shooting",
        # "https://www.youtube.com/watch?v=5QW0Wf7n1F0&list=PL3HxEb3s-wfavO3Wfw-9jeap43HwdTbWR&pp=gAQB":"stealing",
        # "https://www.youtube.com/watch?v=GyAJvaEUUpM&list=PL3HxEb3s-wfYZ_kGn_xwvvDCsYX3HL97a&pp=gAQB":"shoplifting",
        # "https://www.youtube.com/watch?v=AKOnSEDDzDM&list=PL3HxEb3s-wfYZRfCcZnzcpdN-5L77N_Mn&pp=gAQB":"suspicious",
        # "https://www.youtube.com/watch?v=H9hq3onxf0o&list=PL3HxEb3s-wfbCqq76rFlriL45Og2Z8WDT&pp=gAQB":"vandalism",
        # "https://www.youtube.com/watch?v=84lYjtCfIvY&list=PL3HxEb3s-wfZGi6UK_FYPdlvMO7KYxeBl&pp=gAQB":"normal"
    }

    base_folder = "/home/trong/Downloads/Dataset/VAD/origin/crawled_data"
    os.makedirs(base_folder, exist_ok=True)

    ydl_opts = {
        "format": "bv*+ba/best",
        "merge_output_format": "mp4",
        # "cookiefile": "www.youtube.com_cookies.txt",
        "cookiesfrombrowser": ("chrome", ),
        "yes_playlist": True,  # Tải toàn bộ danh sách phát
        "noplaylist": False,  # Không giới hạn chỉ tải video đầu
    }

    for video_url, action_folder in video_lst.items():
        action_path = os.path.join(base_folder, action_folder)
        os.makedirs(action_path, exist_ok=True)

        video_id = 1  # Đếm lại từ 000001 cho mỗi thư mục

        # Lấy danh sách video trong playlist
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)  # Chỉ lấy thông tin, chưa tải
            if "entries" in info_dict:  # Nếu là playlist, lấy danh sách video
                video_entries = info_dict["entries"]
            else:
                video_entries = [info_dict]  # Nếu chỉ có 1 video

        # Tải từng video trong playlist
        for entry in video_entries:
            id_str = f"{video_id:06d}"  # Định dạng số thứ tự thành 6 chữ số
            output_filename = f"crawled_{action_folder}_{id_str}.mp4"
            ydl_opts["outtmpl"] = os.path.join(action_path, output_filename)

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([entry["webpage_url"]])

            video_id += 1  # Tăng ID sau khi tải xong mỗi video

    print("Tất cả video đã được tải xong và lưu vào thư mục tương ứng!")
    return None


if __name__ == '__main__':
    main()
