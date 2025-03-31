import click
import numpy as np
import OpenEXR
import Imath
from PIL import Image
import os
import time
from scipy.ndimage import convolve

def load_exr_grayscale(path: str) -> np.ndarray:
    exr = OpenEXR.InputFile(path)
    dw = exr.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels = exr.header()['channels']
    channel_name = 'R'
    if channel_name not in channels:
        raise ValueError("EXRに 'R' チャンネルが見つかりません")

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    raw = exr.channel(channel_name, pt)
    arr = np.frombuffer(raw, dtype=np.float32)
    arr = arr.reshape((height, width))

    return arr

def load_image_grayscale(path: str) -> np.ndarray:
    img = Image.open(path)
    arr = np.array(img, dtype=np.float32)

    if img.mode in ("I;16", "I;16B", "I;16L"):
        arr /= 65535.0  # 16bit画像は0〜1に正規化

    return arr

def load_heightmap(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".exr":
        return load_exr_grayscale(path)
    else:
        return load_image_grayscale(path)

def save_vdm_exr(vdm: np.ndarray, path: str) -> None:
    assert vdm.ndim == 3 and vdm.shape[2] == 3, "vdm must be HxWx3 array"
    h, w, _ = vdm.shape

    header = OpenEXR.Header(w, h)
    float_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = {
        'R': float_chan,
        'G': float_chan,
        'B': float_chan,
    }

    R = vdm[:, :, 0].astype(np.float32).tobytes()
    G = vdm[:, :, 1].astype(np.float32).tobytes()
    B = vdm[:, :, 2].astype(np.float32).tobytes()

    exr = OpenEXR.OutputFile(path, header)
    exr.writePixels({'R': R, 'G': G, 'B': B})
    exr.close()

def compute_vdm_laplacian(
    heightmap: np.ndarray,
    iterations: int = 1,
    lambda_factor: float = 0.2,
    use_8_neighbors: bool = False,
    preserve_volume: bool = False,
    use_normalized: bool = False,
    mask: np.ndarray | None = None
) -> np.ndarray:
    h, w = heightmap.shape
    current = heightmap.copy()
    vdm = np.zeros((h, w, 3), dtype=np.float32)

    if use_normalized:
        # 広めの正規化カーネル（5x5）
        kernel = np.array([
            [1/np.sqrt(8), 1/np.sqrt(5), 1.0, 1/np.sqrt(5), 1/np.sqrt(8)],
            [1/np.sqrt(5), 1/np.sqrt(2), 1.0, 1/np.sqrt(2), 1/np.sqrt(5)],
            [1.0,          1.0,         0.0, 1.0,          1.0         ],
            [1/np.sqrt(5), 1/np.sqrt(2), 1.0, 1/np.sqrt(2), 1/np.sqrt(5)],
            [1/np.sqrt(8), 1/np.sqrt(5), 1.0, 1/np.sqrt(5), 1/np.sqrt(8)]
        ], dtype=np.float32)
        kernel /= kernel.sum()
        kernel /= kernel.sum()
    else:
        if use_8_neighbors:
            kernel = np.array([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]], dtype=np.float32)
            kernel = kernel / 8.0
        else:
            kernel = np.array([[0, 1, 0],
                               [1, 0, 1],
                               [0, 1, 0]], dtype=np.float32)
            kernel = kernel / 4.0

    if mask is None:
        mask = np.ones_like(heightmap, dtype=np.float32)

    for _ in range(iterations):
        avg = convolve(current, kernel, mode='reflect')
        laplacian = avg - current
        if preserve_volume:
            laplacian *= -1
        current += lambda_factor * laplacian * mask

    dx = (np.roll(current, -1, axis=1) - np.roll(current, 1, axis=1)) * 0.5
    dy = (np.roll(current, -1, axis=0) - np.roll(current, 1, axis=0)) * 0.5
    dz = heightmap  # current - heightmap

    vdm[:, :, 0] = dx
    vdm[:, :, 1] = dy
    vdm[:, :, 2] = dz

    return vdm

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--iterations", "-i", default=1, help="平滑化の繰り返し回数")
@click.option("--lambda-factor", "-l", default=0.2, help="平滑化強度（正：なだらか／負：シャープ）")
@click.option("--use-8-neighbors", "-n", is_flag=True, help="8近傍を使う（デフォルトは4近傍）")
@click.option("--preserve-volume", "-v", is_flag=True, help="体積を保持する（膨張を防ぐ）")
@click.option("--use-normalized", "-u", is_flag=True, help="距離正規化した重み付き平均を使う")
def cli(input_path, output_path, iterations, lambda_factor, use_8_neighbors, preserve_volume, use_normalized):
    print(f"読み込み: {input_path}")
    heightmap = load_heightmap(input_path)

    print("処理開始")
    start = time.time()

    vdm = compute_vdm_laplacian(
        heightmap,
        iterations=iterations,
        lambda_factor=lambda_factor,
        use_8_neighbors=use_8_neighbors,
        preserve_volume=preserve_volume,
        use_normalized=use_normalized
    )

    elapsed = time.time() - start
    print(f"処理完了（{elapsed:.2f}秒）")

    print(f"保存: {output_path}")
    save_vdm_exr(vdm, output_path)

if __name__ == "__main__":
    cli()
