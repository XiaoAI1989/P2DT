# 初始化一个字典来存储每个环境的最大值
max_eval_mean = {}
date = "medium"
time = "10-36-48"
filepath = f"./runs/{date}/{time}_/main.log"

REF_MAX_SCORE = {
    'HalfCheetah': 12135.0,
    'Walker2d': 4592.3,
    'Hopper': 3234.3,
}

REF_MIN_SCORE = {
    'HalfCheetah': -280.178953,
    'Walker2d': 1.629008,
    'Hopper': -20.272305,
}
HalfCheetah_list = [0]
Hopper_list = [0]
Walker2d = [0]


def get_d4rl_normalized_score(env_name, score):
    assert env_name in REF_MAX_SCORE, f'no reference score for {env_name} to calculate d4rl score'
    return (score - REF_MIN_SCORE[env_name]) / (REF_MAX_SCORE[env_name] - REF_MIN_SCORE[env_name]) * 100


# 打开文件并读取每一行

print("HalfCheetah: ", get_d4rl_normalized_score("HalfCheetah", -136.14))
print("Hopper: ", get_d4rl_normalized_score("Hopper", 1179.08, ))
print("Walker2d: ", get_d4rl_normalized_score("Walker2d", 3663.1195))
