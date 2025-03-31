import click
from scipy.ndimage import label, gaussian_filter
import numpy as np
import time
from util import (load_heightmap, save_vdm_exr, height_to_vdm,
                   create_base_mesh, apply_vdm, update_vdm_with_difference)


# メッシュブラー
def blur_mesh(
    mesh: np.ndarray,
    sigma: float = 4.0,
) -> None:
    # mesh[..., 0] = gaussian_filter(mesh[..., 0], sigma=sigma)
    # mesh[..., 1] = gaussian_filter(mesh[..., 1], sigma=sigma)
    mesh[..., 2] = gaussian_filter(mesh[..., 2], sigma=sigma)


# メッシュ変形１
def shrink_positive_z_groups(
    mesh: np.ndarray,
    a: float = 0.1,
    b: float = 4.0,
    gamma: float = 1.0
) -> None:
    """
    Z変位が正の領域をグループ化し、各グループをXY重心中心に縮小する。

    Parameters:
        mesh (np.ndarray): 入力メッシュ (H, W, 3)
        a (float): Z=0のときの縮小率
        b (float): Z=1のときの縮小率
        gamma (float): 補間のガンマ係数（デフォルト: 1.0 で直線）

    Returns:
        np.ndarray: 変形後のメッシュ (H, W, 3)
    """
    z_positive_mask = mesh[..., 2] > 0
    structure = np.ones((3, 3), dtype=bool)  # 8近傍
    labeled, num_labels = label(z_positive_mask, structure=structure)
    for label_id in range(1, num_labels + 1):
        mask = labeled == label_id
        indices = np.argwhere(mask)

        if len(indices) == 0:
            continue

        # 該当点の座標を取得
        points = mesh[mask]
        z_values = points[:, 2]
        xy_points = points[:, :2]

        # XY重心
        center = xy_points.mean(axis=0)

        # 補間された縮小率をZごとに計算
        t = np.clip(z_values, 0.0, 1.0)
        weights = a * (1 - t**gamma) + b * (t**gamma)  # 補間

        # XY位置を中心から縮小
        new_xy = center + (xy_points - center) * weights[:, np.newaxis]
        mesh[mask, 0] = new_xy[:, 0]
        mesh[mask, 1] = -new_xy[:, 1]


# def compute_vdm_laplacian(
#     heightmap: np.ndarray,
#     iterations: int = 1,
#     lambda_factor: float = 0.2,
#     use_8_neighbors: bool = False,
#     preserve_volume: bool = False,
#     use_normalized: bool = False,
#     mask: np.ndarray | None = None,
#     use_step1: bool = False,
#     use_step2: bool = False,
#     use_step3: bool = False
# ):
#     heightmap: np.ndarray,
#     iterations: int = 1,
#     lambda_factor: float = 0.2,
#     use_8_neighbors: bool = False,
#     preserve_volume: bool = False,
#     use_normalized: bool = False,
#     mask: np.ndarray | None = None
# ) -> np.ndarray:
#     h, w = heightmap.shape
#     if use_step1:
#         return height_to_vdm(heightmap)
#
#     current = heightmap.copy()
#     vdm = np.zeros((h, w, 3), dtype=np.float32)
#
#     if use_normalized:
#         # 広めの正規化カーネル（5x5）
#         kernel = np.array([
#             [1/np.sqrt(8), 1/np.sqrt(5), 1.0, 1/np.sqrt(5), 1/np.sqrt(8)],
#             [1/np.sqrt(5), 1/np.sqrt(2), 1.0, 1/np.sqrt(2), 1/np.sqrt(5)],
#             [1.0,          1.0,         0.0, 1.0,          1.0         ],
#             [1/np.sqrt(5), 1/np.sqrt(2), 1.0, 1/np.sqrt(2), 1/np.sqrt(5)],
#             [1/np.sqrt(8), 1/np.sqrt(5), 1.0, 1/np.sqrt(5), 1/np.sqrt(8)]
#         ], dtype=np.float32)
#         kernel /= kernel.sum()
#     else:
#         if use_8_neighbors:
#             kernel = np.array([[1, 1, 1],
#                                [1, 0, 1],
#                                [1, 1, 1]], dtype=np.float32)
#             kernel = kernel / 8.0
#         else:
#             kernel = np.array([[0, 1, 0],
#                                [1, 0, 1],
#                                [0, 1, 0]], dtype=np.float32)
#             kernel = kernel / 4.0
#
#     if mask is None:
#         mask = np.ones_like(heightmap, dtype=np.float32)
#
#     for _ in range(iterations):
#         avg = convolve(current, kernel, mode='reflect')
#         laplacian = avg - current
#         if preserve_volume:
#             laplacian *= -1
#         current += lambda_factor * laplacian * mask
#
#         # スムージング処理（step2）
#     if use_step2:
#         blur_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
#         blur_kernel /= blur_kernel.sum()
#         current = convolve(current, blur_kernel, mode='reflect')
#
#     dx = (np.roll(current, -1, axis=1) - np.roll(current, 1, axis=1)) * 0.5
#     dy = (np.roll(current, -1, axis=0) - np.roll(current, 1, axis=0)) * 0.5
#     dz = current  # - heightmap
#
#     vdm[:, :, 0] = dx
#     vdm[:, :, 1] = dy
#     vdm[:, :, 2] = dz
#
#     # step3: VDM適用後の座標に基づいてXY方向ラプラシアンを追加
#     if use_step3:
#         pos_x = np.arange(w)[None, :] + vdm[:, :, 0]
#         pos_y = np.arange(h)[:, None] + vdm[:, :, 1]
#         pos_z = vdm[:, :, 2]
#         pos = pos_z  # Zだけを見る
#         blur = convolve(pos, kernel, mode='reflect')
#         diff = (blur - pos) * lambda_factor
#         vdm[:, :, 2] += diff
#
#     return vdm

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--iterations", "-i", default=1, help="平滑化の繰り返し回数")
@click.option("--lambda-factor", "-l", default=0.2, help="平滑化強度（正：なだらか／負：シャープ）")
@click.option("--use-8-neighbors", "-n", is_flag=True, help="8近傍を使う（デフォルトは4近傍）")
@click.option("--preserve-volume", "-v", is_flag=True, help="体積を保持する（膨張を防ぐ）")
@click.option("--use-normalized", "-u", is_flag=True, help="距離正規化した重み付き平均を使う")
@click.option("--no-smooth", is_flag=True, help="XY成分のギザギザ緩和を無効にする")
@click.option("--no-laplacian", is_flag=True, help="VDM適用後のラプラシアン追加を無効にする")
def cli(input_path, output_path, iterations, lambda_factor, use_8_neighbors, preserve_volume, use_normalized, no_smooth, no_laplacian):

    print(f"読み込み: {input_path}")
    height_map = load_heightmap(input_path)

    print("処理開始")
    start = time.time()

    initial_vdm = height_to_vdm(height_map)  # R&Gチャンネルは0 Bチャンネルは0~1
    height, width, _ = initial_vdm.shape

    # ベースメッシュ作成
    base_mesh = create_base_mesh((height, width))

    # 初期VDMを適用して最初の変形メッシュを生成
    displaced_mesh = apply_vdm(base_mesh, initial_vdm)

    # 編集処理
    edited_mesh = displaced_mesh.copy()
    blur_mesh(edited_mesh)
    shrink_positive_z_groups(edited_mesh)

    # 新しいVDMの生成（差分を加算）
    updated_vdm = update_vdm_with_difference(initial_vdm, base_mesh, edited_mesh)

    # if not (no_smooth and no_laplacian):
    #     vertices = compulte_vertices(vdm_map)
    #     vertices_work = copy_vertices(vertices)
    #
    #     if not no_smooth:
    #         vertices_work = smooth_xy(vertices_work)
    #
    #     if not no_laplacian:
    #         vertices_work = compute_vdm_laplacian(vertices_work)
    #
    #     diff = diff_vertices(vertices, vertices_work)
    #
    #     vdm_map = apply_vdm_map_diff(vdm_map, diff)

    elapsed = time.time() - start
    print(f"処理完了（{elapsed:.2f}秒）")

    print(f"保存: {output_path}")
    save_vdm_exr(updated_vdm, output_path)


if __name__ == "__main__":
    cli()
