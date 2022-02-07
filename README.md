# 【SIGNATE】医学論文自動仕分けコンペ解法

[SIGNATE医学論文自動仕分けコンテスト](https://signate.jp/competitions/471)で行われたコンテストで作成したモデル。

## 制作背景・目的
論文のタイトルおよび抄録のテキストデータを用いて、システマティックレビューの対象となる文献か否か（2値）を判定するアルゴリズムを作成するコンテストが開催された。

システマティック・レビューとは、ライフサイエンス、特に医学分野において浸透している研究方法で、特定の研究テーマに対する文献をくまなく調査し、各研究データのバイアスを評価しながら、体系的に同質の研究データを収集・解析する研究手法のことを指す。

本コンテストの中で、システマティックレビューの省略化を目指して、収集された論文の中から目的の論文を「選別」するための機械学習アルゴリズムの構築を行った。

引用：https://signate.jp/competitions/471

## 使用言語・環境
* マシン：Google Colab Pro+
    * NVIDIA Tesla P100
* OS：Ubuntu 18.04.5 LTS (Bionic Beaver)
* 言語：Python3.7.12 / Pytorch
* ライブラリ：[requirements.txt](https://drive.google.com/file/d/1pxEX5DwR1c3lYI7qmj8IipgNf9RsaWP6/view?usp=sharing)


## プログラムの概要
以下に取り組み方針を示す。まず、本コンペのフォーラムでYoshio Sugiyamaさんが公開して下さった[ベースライン](https://signate.jp/competitions/471/discussions/pytorch-bert)を参考にしている。


## 工夫点・モデリングの特徴

### 工夫点
基本方針はBERTを用いた層化K分割交差検証(K=10)による学習と予測のみ。
主な工夫点は以下の3つ。

* BERT系の事前学習済みモデルである`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`を利用
* 層化K分割交差検証(K=10)により、10個のモデルで学習、最終的な予測値をもとめる
* 閾値の調整

### 他取り組んだこと
* BERT・RoBERTa・SciBERT・BlueBERT
* epochs・max_length・Foldの変更
* 記号・stopwordsの削除、大文字を小文字に統一
* PubMedBERTのEmbeddingsをConv1Dで2回畳み込みを行うことでFine-tuning
* データの水増し(逆翻訳)

## プログラムについて
* config.py
  * ファイル出力やモデル学習のためのパラメータ設定用のクラス
* utils.py
  * seed値やloggerなどのツールのクラスや関数
* preprocessing.py
  * データの前処理や読み込みを行うためのクラスや関数
* models.py
  * BERT定義用のクラス
* predictor.py
  * 単一のBERTによる予測を行う汎用クラス
* trainer.py
  * 単一のBERTによる訓練を行う汎用クラス
* main.py
  * `models.py`,`predictor.py`, `trainer.py`, `preprocessing`を用いた医学論文分類器の定義と、それを用いて学習から予測までを行うメインファイル

## 自己評価・感想
* 完成度80%
* 公開ベースラインを参考にしたが、BERTを用いた分類モデルの構築方法や手法について知ることが出来た。また、コンテスト内の順位も上位に入ることができて、取り組みとしてはとても満足した。スケジューラや損失関数・最適化関数、モデルの改修など試したいアイディアはあるので、機会があれば、それらを試用したモデル構築も行ってみたい。
* 出来る限り再利用できるように機能ごとにクラスを分離してプログラムを作成した。今後のために依然として、改善の余地は残っているので、適宜修正しながら資産として残していきたい。

## 手順
* GoogleColabで`test.ipynb`を立ち上げ後、コードを順に実行する。
* なお、GoogleColabの無料環境用に設定パラメータを調整している。

* 事前準備
  * SIGNATEで提供されているデータを用いるのでアカウント登録後、データをダウンロードする必要がある。https://signate.jp/competitions/471/data
  * ダウンロード後、`data/`以下にtrain.csv, test.csv, sample_submit.csvを置く

## 参考文献・URL
* 医学論文の自動仕分けチャレンジ | SIGNATE - Data Science Competition,https://signate.jp/competitions/471
