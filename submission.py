# utils/submission.py
import pandas as pd

def create_submission_file(image_ids, angles, speeds, filename):
    submission_df = pd.DataFrame({
        'image_id': image_ids,
        'angle': angles,
        'speed': speeds
    })
    submission_df.to_csv(filename, index=False)
