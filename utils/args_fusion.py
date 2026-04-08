import argparse
from Arguments.run_args import get_run_args
from Arguments import TimeCAP_args


def merge_args():
    model_args_dict = {
        'TimeCAP': TimeCAP_args,
    }

    # Run hyper-parameters
    parser1 = get_run_args()
    run_args = parser1.parse_args()

    # Model hyper-parameters
    model_args_file = model_args_dict[run_args.model]
    parser2 = model_args_file.get_args()
    model_args = parser2.parse_args()

    # Merged hyper-parameters
    merged_parser = argparse.ArgumentParser(description='Merged parser')

    # 获取 parser1 和 parser2 的参数定义
    parser1_actions = parser1._actions
    parser2_actions = parser2._actions

    added_args = set()

    # Merge model hyper-parameters (model priority run)
    for action in parser2_actions:
        if action.dest == 'help':
            continue
        if action.dest not in added_args:
            merged_parser.add_argument(*action.option_strings,
                                       dest=action.dest,
                                       type=action.type,
                                       default=action.default,
                                       help=action.help,
                                       choices=action.choices)
            added_args.add(action.dest)

    # Merge run hyper-parameters
    for action in parser1_actions:
        if action.dest == 'help':
            continue
        if action.dest not in added_args:
            merged_parser.add_argument(*action.option_strings,
                                       dest=action.dest,
                                       type=action.type,
                                       default=action.default,
                                       help=action.help,
                                       choices=action.choices)
            added_args.add(action.dest)

    # 解析合并后的参数
    args = merged_parser.parse_args()

    return args, run_args, model_args
