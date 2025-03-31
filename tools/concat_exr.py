import click
import numpy as np
import OpenEXR
import Imath
from PIL import Image
import os

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
    img = Image.open(path).convert("F")
    arr = np.array(img, dtype=np.float32)
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

    offsets_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    offsets_8 = offsets_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    offsets = offsets_8 if use_8_neighbors else offsets_4

    if mask is None:
        mask = np.ones_like(heightmap, dtype=np.float32)

    for _ in range(iterations):
        updated = current.copy()
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                center = current[y, x]
                weights = []
                values = []
                for dx, dy in offsets:
                    nx, ny = x + dx, y + dy
                    val = current[ny, nx]
                    if use_normalized:
                        dist = np.sqrt(dx ** 2 + dy ** 2)
                        weight = 1.0 / dist if dist > 0 else 0.0
                        weights.append(weight)
                        values.append(val * weight)
                    else:
                        values.append(val)

                avg = sum(values) / sum(weights) if use_normalized else np.mean(values)
                laplacian = avg - center
                if preserve_volume:
                    laplacian *= -1
                updated[y, x] += lambda_factor * laplacian * mask[y, x]
        current = updated

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            dx = (current[y, x + 1] - current[y, x - 1]) * 0.5
            dy = (current[y + 1, x] - current[y - 1, x]) * 0.5
            dz = current[y, x] - heightmap[y, x]
            vdm[y, x] = [dx, dy, dz]

    return vdm

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--iterations", default=1, help="平滑化の繰り返し回数")
@click.option("--lambda-factor", default=0.2, help="平滑化強度（正：なだらか／負：シャープ）")
@click.option("--use-8-neighbors", is_flag=True, help="8近傍を使う（デフォルトは4近傍）")
@click.option("--preserve-volume", is_flag=True, help="体積を保持する（膨張を防ぐ）")
@click.option("--use-normalized", is_flag=True, help="距離正規化した重み付き平均を使う")
def cli(input_path, output_path, iterations, lambda_factor, use_8_neighbors, preserve_volume, use_normalized):
    print(f"読み込み: {input_path}")
    heightmap = load_heightmap(input_path)

    vdm = compute_vdm_laplacian(
        heightmap,
        iterations=iterations,
        lambda_factor=lambda_factor,
        use_8_neighbors=use_8_neighbors,
        preserve_volume=preserve_volume,
        use_normalized=use_normalized
    )

    print(f"保存: {output_path}")
    save_vdm_exr(vdm, output_path)

if __name__ == "__main__":
    cli()
