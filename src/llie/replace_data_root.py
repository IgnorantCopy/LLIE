import os
import glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Replace data root in LLIE dataset')
    parser.add_argument('--root', type=str, required=True, help='New data root')
    return parser.parse_args()


def replace_data_root(old_root, new_root):
    method_dirs = glob.glob("./configs/*")
    print(len(method_dirs))
    if new_root[-1] == '/':
        new_root = new_root[:-1]
    for method_dir in method_dirs:
        config_files = glob.glob(os.path.join(method_dir, "*.yaml"))
        for config_file in config_files:
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            content = content.replace(old_root, new_root)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(content)


if __name__ == '__main__':
    args = parse_args()
    new_root = args.root
    old_root = "/home/nju-student/mkh/datasets/LLIE"
    replace_data_root(old_root, new_root)