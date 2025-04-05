# Height to vdm

It is a simple tool that takes a heightmap and generates a VDM texture.

ハイトマップ（アルファテクスチャ）をベクターディスプレイスメントマップに変換します。
テーパーづけやサイドのスムージングを行い、綺麗ですこしリッチなベクターディスプレイスメントマップを生成します。

## Usage

```
Usage: cli.py [OPTIONS] INPUT_PATH OUTPUT_PATH

Options:
  -ss, --shrink-scales <FLOAT FLOAT>...
                                  Shrink scales（Z0, Z1） / シュリンクスケール（Z0, Z1）
  --gamma FLOAT                   Gamma coefficient for interpolate /
                                  シュリンク補間のガンマ係数
  --no-smooth                     Disable smoothing of XY components /
                                  XY成分のギザギザ緩和を無効にする
  --no-shrink                     Disable Z shrink processing / Zシュリンク処理を無効にする
  --help                          Show this message and exit.
```

## Example

```
python src/height2vdm/cli.py -ss 0.5 1.0 --gamma 2.2 input.png output.vdm
```

Create a VDM with a taper by setting the XY scale to 0.5 times for low Z and 1.0 times for high Z.

Zが低い部分のXYスケールを0.5倍、Zが高い部分のテーパーをXYスケールを1.0倍にして、テーパーのあるVDMを生成します。 

## batch convert

Convert all tif (and other formats) images in a folder to VDMs.

あるフォルダ内のtif（＋その他フォーマット）画像全てをハイトマップとして読み込みしてフォルダに一括コンバートします。

```
./batch_convert.sh -i input_dir -o output_dir -e tif
```

