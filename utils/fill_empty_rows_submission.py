"""Script for filling empty or less-than-five-ind rows in submission"""


from pathlib import Path

import pandas as pd

if __name__ == '__main__':
    submission_path = Path(
        '/home/vadim-tsitko/Projects/SERVER/whales-reid/results/submission_part.csv'
    )
    sample_submission_path = Path('/home/vadim-tsitko/Data/whl/sample_submission.csv')

    submission = pd.read_csv(submission_path)
    sample_submission = pd.read_csv(sample_submission_path)

    print(sample_submission.head())

    sample_submission_part = sample_submission.loc[
        ~sample_submission['image'].isin(submission['image'])
    ]
    print(sample_submission_part.shape)

    res_submission = pd.concat([submission, sample_submission_part])
    print(res_submission.shape)
    print(sample_submission.shape)

    res_submission.to_csv('results/submission_right.csv', index=False)
