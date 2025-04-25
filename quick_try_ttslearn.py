import numpy as np
from ttslearn.tacotron import Tacotron2TTS
import soundfile as sf
import matplotlib.pyplot as plt

def main():
    # モデルの初期化
    model = Tacotron2TTS()
    
    # キーボードからテキストを入力
    text = input("音声に変換したいテキストを入力してください: ")
    
    # テキストをUTF-8にエンコード
    text = text.encode('utf-8').decode('utf-8')
    
    # 音声合成の実行
    wav = model.tts(text)
    
    # tupleの場合は最初の要素を取得
    if isinstance(wav, tuple):
        wav = wav[0]  # 最初の要素を取得
    
    if len(wav.shape) > 1:
        wav = wav.flatten()
    
    # 音声の保存
    sf.write("output.wav", wav, model.sample_rate)
    print("音声ファイルを output.wav として保存しました。")
    
    # 波形の表示
    plt.figure(figsize=(12, 4))
    plt.plot(wav)
    plt.title("voice waveform")
    plt.xlabel("sample")
    plt.ylabel("amplitude")
    plt.grid(True)
    #plt.show()

if __name__ == "__main__":
    main()