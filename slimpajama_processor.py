import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d',
        '--dataset',
        default='DKYoon/SlimPajama-6B',
        type=str,
        help=
        'The dataset to load (Only for SlimPajama series datasets). Default is DKYoon/SlimPajama-6B.',
    )
    parser.add_argument(
        '-s',
        '--split',
        default='train',
        type=str,
        help='The split of the dataset to load. Default is train.',
    )
    parser.add_argument(
        '-r',
        '--ratio',
        default=0.01,
        type=float,
        help='The ratio of the dataset to load. Default is 0.01.',
    )
    parser.add_argument(
        '-op',
        '--output_path',
        type=str,
        required=True,
        help='The directory to save the processed dataset.',
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    print(f'Loading dataset: {args.dataset}; split: {args.split}; ratio: {args.ratio}')

    dataset = load_dataset(
        path=args.dataset,
        split=args.split,
    )

    subset_dict = {}
    for data in tqdm(
            iterable=dataset,
            desc='[Generating Subsets]',
            dynamic_ncols=True,
    ):
        subset_name = data['meta']['redpajama_set_name']
        if subset_name not in subset_dict.keys():
            subset_dict[subset_name] = []

        subset_dict[subset_name].append({
            'content': data['text'],
            'source': subset_name,
            'index': data['__index_level_0__'],
        })

    for subset_name, subset in tqdm(
            iterable=subset_dict.items(),
            desc='[Saving Subsets]',
            dynamic_ncols=True,
    ):
        subset_size = int(len(subset) * args.ratio)
        subset = subset[:subset_size]

        os.makedirs(
            name=args.output_path,
            exist_ok=True,
        )

        output_file = os.path.join(
            args.output_path,
            f'{subset_name}.jsonl',
        )
        with open(
                file=output_file,
                mode='w',
                encoding='utf-8',
        ) as f:
            for item in subset:
                f.write(json.dumps(
                    obj=item,
                    ensure_ascii=False,
                ) + '\n')

        print(
            f'Processed {subset_name} with {len(subset)} items, saved to {output_file}.'
        )
