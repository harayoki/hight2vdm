#!/bin/bash

# 使用方法の表示
usage() {
    echo "Usage: $0 -i INPUT_DIR -o OUTPUT_DIR [-e EXT]"
    echo "  -i INPUT_DIR : 入力画像フォルダ"
    echo "  -o OUTPUT_DIR : 出力フォルダ"
    echo "  -e EXT : 拡張子 (デフォルト: tif)"
    exit 1
}

# デフォルト値
EXT="tif"

# オプション解析
while getopts "i:o:e:" opt; do
    case "$opt" in
        i) INPUT_DIR="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        e) EXT="$OPTARG" ;;
        *) usage ;;
    esac
done

# 必須引数チェック
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    usage
fi

# 出力フォルダがなければ作成
mkdir -p "$OUTPUT_DIR"

# Python スクリプトのパス
SCRIPT="python src/height2vdm/cli.py"

# 入力ファイルを処理
for INPUT_FILE in "$INPUT_DIR"/*.$EXT; do
    # 入力ファイルが存在しない場合スキップ
    [ -e "$INPUT_FILE" ] || continue

    # ベース名取得
    BASENAME=$(basename "$INPUT_FILE" .$EXT)

    # 出力ファイル名
    OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}_vdm.exr"

    echo "Processing: $INPUT_FILE -> $OUTPUT_FILE"

    # Python スクリプト実行
    $SCRIPT "$INPUT_FILE" "$OUTPUT_FILE"
done

echo "All files processed!"
