import os
import pandas as pd

def collect_metadata(path):
    """
    Collecting metadata(video name, lables) from DFDC dataset.
    """
    rf_dict = {"REAL": 0, "FAKE": 1}
    df = pd.read_json(os.path.join(path + '/metadata.json')).T
    df.drop(columns=['split', 'original'], inplace=True)
    df['label'] = df['label'].apply(lambda x: rf_dict[x])
    return df

def make_metadf(path):
    # create metadata
    meta = pd.DataFrame()

    for folder in os.listdir(path):
        if folder == 'boxes': continue
        meta = pd.concat([meta, collect_metadata(path + "/" + folder)])

    meta = meta.reset_index()
    print("Process End!!!")
    return meta

def is_fake(meta, folder, data='DFDC'):
    """
    Function that returns whether passed
    video is real or fake.
    original = 0
    fake = 1
    """
    if data == 'DFDC':
        if meta[meta['index']==(folder + '.mp4')]['label'].item() == 1:
            return 1.
        else:
            return 0.
    elif data == "FaceForensics":
        if meta[-1] == folder:
            return 0.
        else:
            return 1.
    elif data == 'celebv2':
        if meta == folder:
            return 1.
        else:
            return 0.