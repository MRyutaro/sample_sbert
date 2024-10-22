# sample_sbert

main.py <file1_path> <file2_path>

main.pyを実行すると、file1_pathとfile2_pathの2つのファイルの類似度を計算します。
file_pathはhtmlであり、htmlの中身をテキストに変換して類似度を計算します。

テストデータ
| ファイル名 | タイトル |
| --- | --- |
| 1.html | 小型センサで植物を見守るスマート農業の新技術を開発クラウド連携でいつでも、どこでも健康状態のモニタリングが可能に |
| 2.html | 有機溶媒を全く用いず分子触媒によるメカノケミカルアンモニア合成に成功！―窒素ガスとセルロースからのアンモニア合成が可能に― |
| 3.html | パルスレーザーで生成した量子もつれ光を用いて高分解能の量子赤外分光を実証―超高速現象を、小型・高感度で観察する新装置へ－ |
| 4.html | 新型コロナウイルス変異株の構造特性を解明―流行株の変化を原子レベルで可視化― |
| 5.html | 人工知能を活用した新しい医療診断システムの開発に成功―早期発見と正確な診断で医療の質を向上― |
| 6.html | 次世代バッテリー技術のブレークスルー―長寿命・高効率なエネルギー貯蔵が可能に― |
| 7.html | 海洋プラスチック問題を解決する新たな分解技術を開発―環境に優しい持続可能なソリューションを提案― |
| 8.html | 宇宙空間での植物育成実験が成功―将来の宇宙農業に向けた重要な一歩― |
| 9.html | 量子コンピュータを用いた新しい暗号技術の開発に成功―次世代の情報セキュリティを確保する技術― |
| 10.html | 再生可能エネルギーを効率的に利用する新型風力発電機の開発―小型で高効率なエネルギー変換を実現― |
| 11.html | 深層学習を活用した画像認識技術の進化―より正確で迅速な物体認識を可能に― |
| 12.html | 自動運転技術の安全性向上に向けた新しいAIアルゴリズムの開発―複雑な交通環境での正確な判断を実現― |
| 13.html | 新しい電気自動車向けモーター技術の開発に成功―軽量化と高効率化を両立する設計― |
| 14.html | 次世代ロボット技術を支える新たなアクチュエータを開発―より自然な動作を実現する革新技術― |
| 15.html | 高性能な光ファイバー通信技術の開発―データ通信速度の飛躍的向上を実現― |
| 16.html | バイオ燃料の新しい製造プロセスを確立―持続可能なエネルギー源としての可能性を広げる― |
| 17.html | 新しい超伝導材料の発見―次世代エレクトロニクスへの応用に期待― |
| 18.html | 高感度センサーを用いた地震予知技術の開発に成功―災害リスクの低減に貢献― |
| 19.html | 次世代5G通信技術を支える新しいアンテナ設計を開発―通信速度と安定性を向上― |
| 20.html | 微生物を利用した新たな水質浄化技術を開発―環境保全に寄与する持続可能なソリューション― |
| 21.html | ナノテクノロジーを用いた新しい医薬品デリバリーシステムの開発に成功―副作用を抑えた効率的な治療を実現― |
| 22.html | 高温超伝導体を用いた新しい発電技術の開発―エネルギー効率の向上に寄与― |
| 23.html | 次世代ディスプレイ技術の開発に成功―より鮮明で低消費電力な表示を実現― |
| 24.html | ドローンを活用した新しい物流システムの実証実験に成功―効率的で迅速な配送を実現― |
| 25.html | AIを用いた気象予測精度の向上に成功―より正確な天候予測で防災に貢献― |
| 26.html | 次世代のエネルギー貯蔵技術としてのフレキシブルバッテリーの開発に成功―多様な応用可能性を探る― |
| 27.html | 新しい水素生成技術の開発―クリーンエネルギー社会の実現に向けた一歩― |
| 28.html | 超音波技術を用いた非破壊検査の精度向上に成功―インフラの安全性を確保― |
| 29.html | 次世代バイオテクノロジーを活用した食料生産技術の革新―持続可能な食糧供給を目指して― |
| 30.html | 宇宙ごみを除去する新たなロボット技術を開発―宇宙環境保全に向けた取り組み― |
| 31.html | 低コストで高性能な太陽電池の開発に成功―再生可能エネルギーの普及に貢献― |
| 32.html | 自然災害時の情報収集に特化した新しいドローンシステムを開発―迅速な災害対応を可能に― |
| 33.html | 次世代の電動航空機技術の開発に成功―環境に優しい航空輸送を目指して― |
| 34.html | 新しい遺伝子編集技術の開発―より正確で効率的な遺伝子操作が可能に― |
| 35.html | 環境に配慮した新しいプラスチック代替素材を開発―持続可能な社会に向けた革新― |
| 36.html | AIを活用した新たな音声認識技術の開発に成功―より自然な音声インターフェースを実現― |
| 37.html | 海洋生態系を保全するための新しい観測技術を開発―持続可能な海洋利用に貢献― |
| 38.html | 次世代の高効率燃料電池技術の開発に成功―クリーンエネルギー社会の実現を加速― |
| 39.html | バーチャルリアリティを用いた新しい教育プログラムの開発―体験型学習で教育の質を向上― |
| 40.html | ロボットによる高精度な医療手術システムの開発に成功―より安全で正確な手術を実現― |
| 41.html | 気候変動に対応するための新しい森林管理技術を開発―持続可能な森林資源の利用を目指して― |
