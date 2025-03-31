import numpy as np
import OpenEXR
import Imath
from PIL import Image
import os
from typing import Tuple


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


def height_to_vdm(heightmap: np.ndarray) -> np.ndarray:
    h, w = heightmap.shape
    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)
    dz = heightmap.astype(np.float32)
    vdm = np.stack([dx, dy, dz], axis=-1)
    return vdm


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


def create_base_mesh(shape: Tuple[int, int]) -> np.ndarray:
    """
    指定されたサイズのXY正規化メッシュを生成する。

    Parameters:
        shape (Tuple[int, int]): (高さ, 幅)

    Returns:
        np.ndarray: 形状 (H, W, 3) のXY平面メッシュ
    """
    height, width = shape
    y_coords, x_coords = np.meshgrid(
        np.linspace(0, 1, height, endpoint=True),
        np.linspace(0, 1, width, endpoint=True),
        indexing='ij'
    )
    base_mesh = np.stack([x_coords, y_coords, np.zeros_like(x_coords)], axis=-1)
    return base_mesh.astype(np.float32)


def apply_vdm(base_mesh: np.ndarray, vdm: np.ndarray) -> np.ndarray:
    """
    ベースメッシュにVDMを適用して変形後メッシュを生成する。

    Parameters:
        base_mesh (np.ndarray): 元のメッシュ (H, W, 3)
        vdm (np.ndarray): ベクターディスプレースメントマップ (H, W, 3)

    Returns:
        np.ndarray: 変形後のメッシュ (H, W, 3)
    """
    return base_mesh + vdm


def update_vdm_with_difference(
    original_vdm: np.ndarray,
    base_mesh: np.ndarray,
    modified_vertices: np.ndarray
) -> np.ndarray:
    """
    元のVDMと、新たな変形の差分を加算して更新版VDMを生成。

    Parameters:
        original_vdm (np.ndarray): 元のVDM (H, W, 3)
        base_mesh (np.ndarray): 元のメッシュ (H, W, 3)
        modified_vertices (np.ndarray): 編集後のメッシュ (H, W, 3)

    Returns:
        np.ndarray: 更新されたVDM (H, W, 3)
    """
    delta = modified_vertices - (base_mesh + original_vdm)
    return original_vdm + delta

