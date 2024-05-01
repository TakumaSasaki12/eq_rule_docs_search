import os
from google.cloud import storage
from PyPDF2 import PdfFileReader

# PDFファイルのパス
pdf_file_path = "path/to/pdf_file.pdf"

# Google Cloud Storageへの接続
client = storage.Client()
bucket_name = "your_bucket_name"
bucket = client.get_bucket(bucket_name)

# PDFファイルをアップロード
blob = bucket.blob("pdf_file.pdf")
blob.upload_from_filename(pdf_file_path)

# PDFファイルの各ページをテキストに変換して保存
with open(pdf_file_path, "rb") as f:
    pdf = PdfFileReader(f)
    for page_num in range(pdf.numPages):
        page = pdf.getPage(page_num)
        text = page.extractText()

        # テキストデータをクラウドストレージに保存
        text_blob = bucket.blob(f"text_files/page_{page_num+1}.txt")
        text_blob.upload_from_string(text, content_type="text/plain")
