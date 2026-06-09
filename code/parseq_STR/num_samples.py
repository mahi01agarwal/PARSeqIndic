from strhub.data.dataset import LmdbDataset
import argparse


parser = argparse.ArgumentParser(description="take lmdb path")
parser.add_argument("--lmdb_root", type=str, help="path to the LMDB root directory")

args = parser.parse_args()
lmdb_root = args.lmdb_root
charset = "0123456789\"',.:;?-_!#$%&()*+/<=>@[]^{|}~\\।॥ਁਂਃਅਆਇਈਉਊਏਐਓਔਕਖਗਘਙਚਛਜਝਞਟਠਡਢਣਤਥਦਧਨਪਫਬਭਮਯਰਲਲ਼ਵਸ਼ਸਹ਼ਾਿੀੁੂੇੈੋੌ੍ੑਖ਼ਗ਼ਜ਼ੜਫ਼੦੧੨੩੪੫੬੭੮੯ੰੱੲੳੴੵ "

max_label_len = 35  # specify the maximum label length allowed
min_image_dim = 0
remove_whitespace = True  # specify if you want to remove whitespace from labels
normalize_unicode = False  # specify if you want to normalize Unicode characters
unlabelled = False  # specify if your dataset is unlabelled

# Create an instance of the LmdbDataset class
lmdb_dataset = LmdbDataset(lmdb_root, charset, max_label_len, min_image_dim,
                           remove_whitespace, normalize_unicode, unlabelled)

# Get the number of samples
num_samples = len(lmdb_dataset)
print("Number of samples:", num_samples)

