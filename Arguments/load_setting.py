def get_setting_str(args):

    setting_map = {
        'TimeCAP': '{}_{}_{}_sl{}',
        'default': '{}_{}',
    }

    format_str = setting_map.get(args.model, setting_map["default"])

    if args.model == 'TimeCAP':
        setting = format_str.format(
            args.task_name,
            args.model,
            args.data,
            args.seq_len,
        )
    else:
        setting = format_str.format(
            args.task_name,
            args.data,
            )

    return setting