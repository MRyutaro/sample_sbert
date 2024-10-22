import sys
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util


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


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <file1_path> <file2_path>")
        sys.exit(1)

    file1_path = sys.argv[1]
    file2_path = sys.argv[2]

    # ファイルからテキストを抽出
    text1 = extract_text_from_html(file1_path)
    text2 = extract_text_from_html(file2_path)

    # 類似度を計算
    similarity_score = calculate_similarity(text1, text2)
    print(f"Similarity score between the two documents: {similarity_score:.4f}")
