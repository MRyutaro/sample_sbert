import sys
import time
import psutil
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from memory_profiler import memory_usage, profile

MODEL_NAME = 'stsb-xlm-r-multilingual'


# HTMLファイルからテキストを抽出する関数
def extract_text_from_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    soup = BeautifulSoup(content, 'html.parser')
    return soup.get_text()


# 類似度計算を行う関数
def calculate_similarity(text1, text2):
    model = SentenceTransformer(MODEL_NAME)
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
    return similarity.item()


# メイン処理
@profile
def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <file1_path> <file2_path>")
        sys.exit(1)

    file1_path = sys.argv[1]
    file2_path = sys.argv[2]

    # CPUおよびメモリ使用率の取得開始
    process = psutil.Process()
    start_cpu_percent = psutil.cpu_percent(interval=None)
    start_memory_info = process.memory_info().rss / (1024 * 1024)  # メモリ使用量（MB）

    # 処理時間の計測開始
    start_time = time.time()

    # ファイルからテキストを抽出
    text1 = extract_text_from_html(file1_path)
    text2 = extract_text_from_html(file2_path)

    # テキストの文字数とデータ量を表示
    text1_size = len(text1.encode('utf-8')) / 1024  # KB単位のデータ量
    text2_size = len(text2.encode('utf-8')) / 1024  # KB単位のデータ量
    print(f"ファイル1の文字数: {len(text1)}文字, データ量: {text1_size:.2f} KB")
    print(f"ファイル2の文字数: {len(text2)}文字, データ量: {text2_size:.2f} KB")

    # 類似度を計算
    similarity_score = calculate_similarity(text1, text2)

    # 処理時間の計測終了
    end_time = time.time()
    elapsed_time = end_time - start_time

    # CPUおよびメモリ使用率の取得終了
    end_cpu_percent = psutil.cpu_percent(interval=None)
    end_memory_info = process.memory_info().rss / (1024 * 1024)  # メモリ使用量（MB）

    # 結果を表示
    print()
    print(f"2つのドキュメント間の類似度スコア: {similarity_score:.4f}")
    print()
    print(f"処理時間: {elapsed_time:.2f} 秒")
    print(f"初期CPU使用率: {start_cpu_percent}%")
    print(f"最終CPU使用率: {end_cpu_percent}%")
    print(f"初期メモリ使用量: {start_memory_info:.2f} MB")
    print(f"最終メモリ使用量: {end_memory_info:.2f} MB")


# メモリ使用量の計測
if __name__ == "__main__":
    mem_usage = memory_usage(main, interval=0.1)
    print(f"実行中のメモリ使用量: {max(mem_usage) - min(mem_usage):.2f} MB")
