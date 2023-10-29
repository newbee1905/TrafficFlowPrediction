import gdown
import concurrent.futures

def extract_file_id(url):
    return url.split("/file/d/")[1].split("/view")[0]

def download_file(url, output):
    file_id = extract_file_id(url)
    download_url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(download_url, output, quiet=False)


lstm = "https://drive.google.com/file/d/1-q31FVAdOYPh_5ylxVhvQ1EMA6IQbaLz/view?usp=drive_link"
gru = "https://drive.google.com/file/d/1-yZTI7qNKO5110fuW28wLQMycWxSkrQS/view?usp=drive_link"
saes = "https://drive.google.com/file/d/1-Qg898nPhPfMoziWBQarg5O-aUYCgRL7/view?usp=drive_link"
cnn = "https://drive.google.com/file/d/1-Frxx7pP-YFmmv5jxgvR7KyNwcA7Hym0/view?usp=drive_link"
prophet = "https://drive.google.com/file/d/1-kZ3d6vwQpEgdBWmepUE3Io1dhJziUKi/view?usp=drive_link"

models = [
  {
    "url": lstm,
    "file": "model/lstm.h5",
  },
  {
    "url": gru,
    "file": "model/gru.h5",
  },
  {
    "url": saes,
    "file": "model/saes.h5",
  },
  {
    "url": cnn,
    "file": "model/cnn.h5",
  },
  {
    "url": prophet,
    "file": "model/prophet.json",
  },
]

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(download_file, model["url"], model["file"]) for model in models]

    for future in concurrent.futures.as_completed(futures):
        future.result()
