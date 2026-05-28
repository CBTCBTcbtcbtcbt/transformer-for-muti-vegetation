import csv
import random
from pathlib import Path

# group_num 表示需要在已有数据后面新增多少组不重复的序列。
# 你可以直接修改这个变量，例如改成 100、1000 等。
group_num = 1

# point_num 表示每一个新生成序列中 1 的数量。
# 例如 point_num = 8，就表示新生成的每一行都有 8 个位置是 1。
point_num = 22

# USE_INPUT_CSV 是一个开关变量。
# 当它为 True 时，程序会先读取 INPUT_CSV_PATH 对应的 CSV 文件，
# 再在读取到的内容后面追加 group_num 组新序列。
# 
USE_INPUT_CSV = True

# USE_INPUT_CSV = False

# INPUT_CSV_PATH 表示可选输入 CSV 文件路径。
# 只有当 USE_INPUT_CSV 为 True 时，这个路径才会被使用。
INPUT_CSV_PATH = Path("input.csv")

# SEQUENCE_LENGTH 表示每一组序列的长度，本任务固定为 37。
SEQUENCE_LENGTH = 37

# OUTPUT_CSV_PATH 表示最终保存 output 列表的 CSV 文件路径。
OUTPUT_CSV_PATH = Path("input.csv")

# FORCE_ONE_INDEXES 使用 0-based 下标，强制这些位置始终为 1。
# 用户要求“1号和37号一定被选中”，对应下标分别是 0 和 36。
FORCE_ONE_INDEXES = (0, 36)

# MAX_FAILED_ATTEMPTS 表示连续多少次随机生成都重复后停止程序。
# 这个保护用于避免 point_num 太小、可用组合已经耗尽时程序无限循环。
MAX_FAILED_ATTEMPTS = 100_000

# rotation_mapping 表示旋转一次时，每个位置上的值会移动到哪个新位置。
# 例如 0: 3 表示原来下标 0 的值，在旋转一次后会放到下标 3。
rotation_mapping = {
    # 第 1 行
    0: 3, 1: 8, 2: 14, 3: 21,
    # 第 2 行
    4: 2, 5: 7, 6: 13, 7: 20, 8: 27,
    # 第 3 行
    9: 1, 10: 6, 11: 12, 12: 19, 13: 26, 14: 32,
    # 第 4 行，包含中心点
    15: 0, 16: 5, 17: 11, 18: 18, 19: 25, 20: 31, 21: 36,
    # 第 5 行
    22: 4, 23: 10, 24: 17, 25: 24, 26: 30, 27: 35,
    # 第 6 行
    28: 9, 29: 16, 30: 23, 31: 29, 32: 34,
    # 第 7 行
    33: 15, 34: 22, 35: 28, 36: 33,
}


def generate_random_group(required_point_num: int) -> list[int]:
    """随机生成一个长度为 37，且正好包含 required_point_num 个 1 的 0/1 列表。"""
    forced_set = set(FORCE_ONE_INDEXES)
    forced_count = len(forced_set)
    if required_point_num < forced_count:
        raise ValueError(
            f"point_num 必须至少为 {forced_count}，因为存在强制选中点位：{FORCE_ONE_INDEXES}。"
        )

    # 先创建一个全是 0 的序列，并把强制点位写成 1。
    group = [0] * SEQUENCE_LENGTH
    for forced_index in forced_set:
        group[forced_index] = 1

    # 其余要补充的 1，从非强制点位中随机抽取。
    remain_ones = required_point_num - forced_count
    optional_indexes = [i for i in range(SEQUENCE_LENGTH) if i not in forced_set]
    sampled_optional = random.sample(optional_indexes, remain_ones)
    for index in sampled_optional:
        group[index] = 1

    # 返回生成好的序列。
    return group


def rotate_once(group: list[int]) -> list[int]:
    """根据 rotation_mapping 将一个序列旋转一次。"""
    # 创建一个长度为 37 的空列表，用来存放旋转后的结果。
    rotated_group = [0] * SEQUENCE_LENGTH

    # 遍历 rotation_mapping 中的每一组映射关系。
    for source_index, target_index in rotation_mapping.items():
        # 把原序列 source_index 位置上的值，放到旋转后序列 target_index 的位置上。
        rotated_group[target_index] = group[source_index]

    # 返回旋转一次后的新序列。
    return rotated_group


def get_rotated_groups(group: list[int]) -> list[list[int]]:
    """生成某个序列旋转 0、1、2、3、4、5 次后的全部结果。"""
    # rotations 用来保存旋转 0 到 5 次的结果。
    rotations = []

    # current_group 表示当前旋转次数下的序列；一开始没有旋转，所以就是原序列。
    current_group = group.copy()

    # 六边形结构旋转 6 次会回到原状态，所以这里只需要记录 0 到 5 次旋转。
    for _ in range(6):
        # 把当前旋转次数对应的序列加入 rotations。
        rotations.append(current_group.copy())

        # 在当前序列基础上继续旋转一次，供下一轮循环使用。
        current_group = rotate_once(current_group)

    # 返回全部旋转结果。
    return rotations


def group_to_key(group: list[int]) -> tuple[int, ...]:
    """把列表形式的序列转换成可放进集合里的元组形式。"""
    # 列表不能直接放进集合，但元组可以，所以这里把 list 转成 tuple。
    return tuple(group)


def get_group_variants(group: list[int]) -> tuple[list[list[int]], list[tuple[int, ...]]]:
    """同时得到一个序列的 6 个旋转结果，以及它们对应的元组键。"""
    # 先生成旋转 0 到 5 次的全部结果。
    rotated_groups = get_rotated_groups(group)

    # 再把每个旋转结果转成元组，方便放进集合里做去重判断。
    rotated_keys = [group_to_key(rotated_group) for rotated_group in rotated_groups]

    # 返回旋转结果和它们的键。
    return rotated_groups, rotated_keys


def load_groups_from_csv(csv_path: Path) -> list[list[int]]:
    """从不带表头的 CSV 文件中读取序列。"""
    # 如果输入文件不存在，就直接报错，避免后面出现难以理解的问题。
    if not csv_path.exists():
        raise FileNotFoundError(f"输入 CSV 文件不存在：{csv_path}")

    # 这里保存从 CSV 中读取到的所有序列。
    loaded_groups = []

    # 使用 utf-8-sig 读取，这样即使文件带 BOM 也能正常解析。
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        # 创建 CSV 读取器，逐行读取数据。
        reader = csv.reader(csv_file)

        # 遍历每一行。row_index 从 1 开始，方便报错时定位问题。
        for row_index, row in enumerate(reader, start=1):
            # 如果这一行是空行，就直接跳过。
            if not row or all(cell.strip() == "" for cell in row):
                continue

            # 每一行必须正好有 37 个值，少了或多了都说明文件格式不对。
            if len(row) != SEQUENCE_LENGTH:
                raise ValueError(
                    f"输入 CSV 第 {row_index} 行长度错误："
                    f"期望 {SEQUENCE_LENGTH} 列，实际 {len(row)} 列。"
                )

            # 把字符串形式的 0/1 转成整数，并去掉单元格两端可能存在的空格。
            try:
                group = [int(cell.strip()) for cell in row]
            except ValueError as exc:
                raise ValueError(
                    f"输入 CSV 第 {row_index} 行包含非整数内容。"
                ) from exc

            # 每个元素都必须是 0 或 1，否则就不符合当前任务要求。
            if any(bit not in (0, 1) for bit in group):
                raise ValueError(
                    f"输入 CSV 第 {row_index} 行包含不是 0/1 的值。"
                )

            # 把这一行加入已读取列表。
            loaded_groups.append(group)

    # 如果整个文件没有读到任何有效数据，也要报错，避免空输入被静默接受。
    if not loaded_groups:
        raise ValueError(f"输入 CSV 文件中没有有效数据：{csv_path}")

    # 返回所有读取到的序列。
    return loaded_groups


def build_state_from_seed(
    seed_groups: list[list[int]],
) -> tuple[list[list[int]], list[list[int]], set[tuple[int, ...]]]:
    """把输入 CSV 里的序列转换成 output、total 和 total_keys。"""
    # output 保存已经接受的原始序列。
    output = []

    # total 保存所有已接受序列的 6 个旋转结果。
    total = []

    # total_keys 保存 total 的元组形式，方便快速判断重复。
    total_keys: set[tuple[int, ...]] = set()

    # 逐组处理输入数据。
    # 这里不会检查输入 CSV 是否满足 point_num，也不会因为输入内部有重复就报错。
    # 输入 CSV 的原始行会被完整保留；total_keys 只用于避免后续新增序列和输入内容重复。
    for group in seed_groups:
        # 先计算这一组的 6 个旋转结果，以及对应的元组键。
        rotated_groups, rotated_keys = get_group_variants(group)

        # 输入 CSV 的行不做 point_num 检查，也不做去重拦截，直接加入 output。
        output.append(group)

        # 把它的 6 个旋转结果全部加入 total。
        total.extend(rotated_groups)

        # 同时更新 total_keys，供后续组继续判断重复。
        total_keys.update(rotated_keys)

    # 返回初始状态。
    return output, total, total_keys


def write_output_to_csv(output: list[list[int]], csv_path: Path) -> None:
    """把最终的 output 列表保存到 CSV 文件中。"""
    # 如果 CSV 文件所在的文件夹不存在，就先创建文件夹。
    if csv_path.parent != Path("."):
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 使用 utf-8-sig 编码写入，方便 Excel 正确识别 CSV 文件。
    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        # 创建 CSV 写入器，用来逐行写入数据。
        writer = csv.writer(csv_file)

        # 逐行写入 output 中的每一组序列。
        # 这里不写表头，也不写 group_index，只保留 37 个 0/1 数值。
        for group in output:
            # 每一行只写入这一组的 37 个 0/1 值。
            writer.writerow(group)


def generate_groups(
    append_group_num: int,
    required_point_num: int,
    output: list[list[int]],
    total: list[list[int]],
    total_keys: set[tuple[int, ...]],
) -> tuple[list[list[int]], list[list[int]], set[tuple[int, ...]]]:
    """不断随机生成序列，直到新增数量达到 append_group_num。"""
    # added_count 记录本次运行已经成功新增了多少组。
    added_count = 0

    # failed_attempts 记录连续随机到重复序列的次数，用于防止极端情况下无限循环。
    failed_attempts = 0

    # 当本次新增数量还没有达到 append_group_num 时，就继续随机生成新序列。
    while added_count < append_group_num:
        # 随机生成一个候选序列。
        candidate_group = generate_random_group(required_point_num)

        # 先计算这个候选序列旋转 0 到 5 次后的全部结果。
        # 如果其中任意一个结果已经在 total 中，就说明它和历史序列属于同一旋转等价类，直接跳过。
        candidate_rotations, candidate_keys = get_group_variants(candidate_group)

        # 只要 6 个旋转结果里有任意一个已经出现过，就说明这个候选序列不能再加入 output。
        if any(candidate_key in total_keys for candidate_key in candidate_keys):
            # 记录一次失败尝试。
            failed_attempts += 1

            # 如果连续失败次数太多，说明当前 point_num 和 group_num 可能让可用组合不够。
            if failed_attempts >= MAX_FAILED_ATTEMPTS:
                raise RuntimeError(
                    "连续随机生成到重复序列的次数过多，程序已停止。"
                    "请尝试减小 group_num，或更换 point_num。"
                )

            # 跳过重复候选，继续下一次随机生成。
            continue

        # 候选序列没有重复时，把它加入最终结果 output。
        output.append(candidate_group)

        # 成功新增一组后，更新新增计数。
        added_count += 1

        # 成功新增后，把连续失败次数清零。
        failed_attempts = 0

        # 按需求把旋转 0、1、2、3、4、5 次的结果全部加入 total。
        total.extend(candidate_rotations)

        # 同步更新 total_keys，保证下一轮可以快速判断重复。
        total_keys.update(candidate_keys)

    # 返回最终选中的 output，以及用于去重记录的 total。
    return output, total, total_keys


def main() -> None:
    """程序入口：生成序列并保存 CSV 文件。"""
    # 检查 group_num 是否是正整数，避免传入 0、负数或其他类型导致逻辑错误。
    if not isinstance(group_num, int) or group_num <= 0:
        raise ValueError("group_num 必须是正整数。")

    # 检查 point_num 是否是合法整数，因为它表示每个新序列中 1 的数量。
    min_point_num = len(set(FORCE_ONE_INDEXES))
    if not isinstance(point_num, int) or not min_point_num <= point_num <= SEQUENCE_LENGTH:
        raise ValueError(
            f"point_num 必须是 {min_point_num} 到 {SEQUENCE_LENGTH} 之间的整数。"
        )

    # 如果开关打开，就先读取输入 CSV 作为种子数据；否则从空数据开始生成。
    if USE_INPUT_CSV:
        # 先读取 CSV 文件中的所有序列。
        seed_groups = load_groups_from_csv(INPUT_CSV_PATH)
    else:
        # 不使用输入文件时，seed_groups 就是空列表。
        seed_groups = []

    # 把种子数据转换成 output、total 和 total_keys。
    output, total, total_keys = build_state_from_seed(seed_groups)

    # 再继续随机生成 group_num 组，并追加在输入 CSV 的内容后面。
    output, total, total_keys = generate_groups(
        group_num,
        point_num,
        output,
        total,
        total_keys,
    )

    # 把最终 output 保存到 CSV 文件。
    write_output_to_csv(output, OUTPUT_CSV_PATH)

    # 打印生成结果，方便在命令行中确认程序执行情况。
    print(f"输入 CSV 读取 {len(seed_groups)} 组序列。")
    print(f"本次新增 {group_num} 组序列。")
    print(f"最终输出 {len(output)} 组序列。")
    print(f"每个新增序列包含 {point_num} 个 1。")
    print(f"total 中共保存 {len(total)} 个旋转结果。")
    print(f"CSV 文件已保存到：{OUTPUT_CSV_PATH.resolve()}")


if __name__ == "__main__":
    main()
