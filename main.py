import os
import csv
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = 'stsb-xlm-r-multilingual'


# HTMLファイルからテキストを抽出する関数
def extract_text_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    soup = BeautifulSoup(content, 'html.parser')
    # 空白を削除してテキストを取得
    text = soup.get_text().replace('\n', '').replace('\r', '').replace('\t', '')
    return text


# 類似度計算を行う関数
def calculate_similarity(text1, text2):
    model = SentenceTransformer(MODEL_NAME)
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
    return similarity.item()


# メイン処理
def main():
    folder_path = './data'
    html_files = [f for f in os.listdir(folder_path) if f.endswith('.html')]
    N = 20
    html_files = html_files[:N]
    print(f'{N}個のHTMLファイルを読み込みます。')

    # CSVファイルにヘッダーを書き込む
    with open('similarity_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['File1', 'File2', 'Similarity Score'])

    # すべてのHTMLファイル間の類似度を計算
    for i in range(len(html_files)):
        for j in range(i + 1, len(html_files)):
            file1_path = os.path.join(folder_path, html_files[i])
            file2_path = os.path.join(folder_path, html_files[j])

            # ファイルからテキストを抽出
            text1 = extract_text_from_html(file1_path)
            text2 = extract_text_from_html(file2_path)
            print("=====text1=====")
            print(text1)
            print("=====text2=====")
            print(text2)

            # 類似度を計算
            similarity_score = calculate_similarity(text1, text2)
            print("=====similarity_score=====")
            print(f'{html_files[i]} vs {html_files[j]}: {similarity_score}')

            # 結果をファイルに逐次書き込む
            with open('similarity_results.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([html_files[i], html_files[j], similarity_score])

    print("全てのファイル間の類似度スコアを similarity_results.csv に保存しました。")


# メモリ使用量の計測
if __name__ == "__main__":
    main()
