import click
from scipy.ndimage import label, gaussian_filter
from scipy.ndimage import binary_erosion
import numpy as np
import time
import os
from util import (load_heightmap, save_vdm_exr, height_to_vdm,
                   create_base_mesh, apply_vdm, update_vdm_with_difference)


# メッシュブラー（グループ周辺だけ）
def blur_mesh_around_groups(
    mesh: np.ndarray,
    mask: np.ndarray,
    sigma: float = 4.0,
) -> None:
    # Z成分だけ処理
    z = mesh[..., 2]
    blurred = gaussian_filter(z * mask, sigma=sigma)
    mask_blur = gaussian_filter(mask.astype(float), sigma=sigma)
    mask_blur[mask_blur == 0] = 1  # ゼロ除算防止
    mesh[..., 2] = blurred / mask_blur


# メッシュ変形１
def shrink_positive_z_groups(
    mesh: np.ndarray,
    a: float = 1.0,
    b: float = 2.0,
    gamma: float = 1.0,
    enable_shrink: bool = True,
) -> np.ndarray:
    """
    Z変位が正の領域をグループ化し、各グループをXY重心中心に縮小する。

    Parameters:
        mesh (np.ndarray): 入力メッシュ (H, W, 3)
        a (float): Z=0のときの縮小率
        b (float): Z=1のときの縮小率
        gamma (float): 補間のガンマ係数（デフォルト: 1.0 で直線）
        enable_shrink (bool): 縮小処理を有効にするかどうか

    Returns:
        np.ndarray: グループマスク (H, W)
    """
    original_z = mesh[..., 2].copy()  # 元のZを保存
    z_positive_mask: np.ndarray = original_z > 0
    z_positive_mask = binary_erosion(z_positive_mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))

    # structure = np.ones((3, 3), dtype=bool)  # 8近傍
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=bool) # 4近傍
    labeled, num_labels = label(z_positive_mask, structure=structure)

    # グループ周辺だけのマスクを作る
    group_mask = np.zeros(mesh.shape[:2], dtype=bool)
    group_mask[z_positive_mask] = True

    if not enable_shrink:
        return group_mask

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
        print(f"グループ {label_id} のXY重心: {center}")

        # 補間された縮小率をZごとに計算
        t = np.clip(z_values, 0.0, 1.0)
        weights = a * (1 - t**gamma) + b * (t**gamma)  # 補間

        # ベクトル計算
        diff = xy_points - center
        # diff[:, 1] *= -1.0  # Y方向だけベクトル反転

        # center中心にweightだけ半径をかける
        new_xy = center + diff * weights[:, np.newaxis]

        mesh[mask, 0] = new_xy[:, 0]
        mesh[mask, 1] = new_xy[:, 1]

    return group_mask

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option(
    "-ss",
    "--shrink-scales",
    type=(float, float),
    default=(1.0, 1.5),
    help="shrink scales（Z0, Z1）",
)
@click.option("--gamma", type=float, default=1.0, help="補間のガンマ係数")
@click.option("--no-smooth", is_flag=True, help="XY成分のギザギザ緩和を無効にする")
@click.option("--no-shrink", is_flag=True, help="Zシュリンク処理を無効にする")
def cli(input_path, output_path, shrink_scales, gamma, no_smooth, no_shrink):

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

    # グループマスクを先に取得
    group_mask = shrink_positive_z_groups(edited_mesh, enable_shrink=False)

    if not no_smooth:
        blur_mesh_around_groups(edited_mesh, group_mask)

    # 必要に応じてシュリンク処理
    if not no_shrink:
        shrink_positive_z_groups(
            edited_mesh,
            a=shrink_scales[0],
            b=shrink_scales[1],
            gamma=gamma,
            enable_shrink=True)

    # 新しいVDMの生成（差分を加算）
    updated_vdm = update_vdm_with_difference(base_mesh, edited_mesh)

    elapsed = time.time() - start
    print(f"処理完了（{elapsed:.2f}秒）")

    if os.path.isdir(output_path):
        # 出力先がディレクトリの場合は、ファイル名を付けて保存
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_path, f"{base_name}_vdm.exr")

    print(f"保存: {output_path}")
    save_vdm_exr(updated_vdm, output_path)


if __name__ == "__main__":
    cli()
